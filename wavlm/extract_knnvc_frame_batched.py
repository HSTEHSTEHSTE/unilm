#!/usr/bin/env python3
"""Extract independent one-frame KNN-VC WavLM features in global GPU batches.

Every WavLM invocation contains only 400-sample windows, which produce one
valid 20-ms feature frame.  Windows from different utterances are packed into
the same batch, so no utterance can influence another through attention.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torchaudio


# KNN-VC Torch Hub imports its bundled ``wavlm`` package.  This checkout has
# a local module with the same name, so remove the local import roots first.
SCRIPT_DIRECTORY = Path(__file__).resolve().parent
PROJECT_DIRECTORY = SCRIPT_DIRECTORY.parent
sys.path = [
    entry for entry in sys.path
    if Path(entry or ".").resolve() not in {SCRIPT_DIRECTORY, PROJECT_DIRECTORY}
]
sys.modules.pop("wavlm", None)

SAMPLE_RATE = 16_000
FEATURE_HOP = 320
RECEPTIVE_FIELD = 400
HIDDEN_SIZE = 1024


@dataclass
class UtteranceWork:
    source: Path
    destination: Path
    windows: torch.Tensor
    features: torch.Tensor
    cursor: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--file-list", type=Path, required=True)
    parser.add_argument("--output-layer", type=int, default=6)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--save-dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--probe-initial-batch", type=int, default=256)
    parser.add_argument("--probe-ceiling-batch", type=int, default=1_048_576)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--progress-every-batches", type=int, default=100)
    args = parser.parse_args()
    if args.output_layer < 1:
        parser.error("--output-layer must be positive")
    if args.probe_initial_batch < 1:
        parser.error("--probe-initial-batch must be positive")
    if args.probe_ceiling_batch < args.probe_initial_batch:
        parser.error("--probe-ceiling-batch must be at least --probe-initial-batch")
    if args.progress_every_batches < 1:
        parser.error("--progress-every-batches must be positive")
    return args


def read_manifest(file_list: Path, input_dir: Path) -> list[Path]:
    if not file_list.is_file():
        raise SystemExit(f"Manifest does not exist: {file_list}")
    paths: list[Path] = []
    seen: set[Path] = set()
    for line_number, line in enumerate(file_list.read_text().splitlines(), start=1):
        relative = Path(line.strip())
        if not line.strip():
            continue
        if relative.is_absolute() or ".." in relative.parts:
            raise SystemExit(f"Invalid relative path on line {line_number}: {line!r}")
        source = (input_dir / relative).resolve()
        try:
            source.relative_to(input_dir)
        except ValueError as error:
            raise SystemExit(f"Manifest path escapes input root on line {line_number}: {line!r}") from error
        if not source.is_file():
            raise SystemExit(f"Missing audio on line {line_number}: {source}")
        if source in seen:
            raise SystemExit(f"Duplicate audio path on line {line_number}: {line!r}")
        paths.append(source)
        seen.add(source)
    return paths


def destination_for(source: Path, input_dir: Path, output_dir: Path) -> Path:
    return (output_dir / source.relative_to(input_dir)).with_suffix(".pt")


def save_tensor_atomically(tensor: torch.Tensor, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        torch.save(tensor, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def load_windows(source: Path) -> torch.Tensor:
    waveform, sample_rate = torchaudio.load(str(source))
    if waveform.numel() == 0:
        raise RuntimeError(f"Empty audio file: {source}")
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sample_rate, SAMPLE_RATE)
    waveform = waveform.to(dtype=torch.float32)
    if waveform.shape[1] < RECEPTIVE_FIELD:
        return torch.empty((0, RECEPTIVE_FIELD), dtype=torch.float32)
    # Each row is exactly one valid convolutional receptive field.  There is
    # no preceding history and no future frame in the sequence passed to WavLM.
    return waveform[0].unfold(0, RECEPTIVE_FIELD, FEATURE_HOP)


def is_cuda_oom(error: RuntimeError) -> bool:
    message = str(error).lower()
    # ``scaled_dot_product_attention`` on this CUDA/PyTorch stack also has a
    # launch-grid limit: batches at or above 65,536 fail with an invalid
    # configuration rather than an OOM.  It is still a real upper bound for
    # this exact extraction kernel, so include it in the capacity search.
    return "out of memory" in message or "invalid configuration argument" in message


def try_batch_size(model, batch_size: int, output_layer: int, device: torch.device) -> bool:
    sources: torch.Tensor | None = None
    features: torch.Tensor | None = None
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        sources = torch.zeros((batch_size, RECEPTIVE_FIELD), device=device)
        with torch.inference_mode():
            features, _ = model.extract_features(sources, output_layer=output_layer)
        torch.cuda.synchronize(device)
        del features, sources
        torch.cuda.empty_cache()
        return True
    except RuntimeError as error:
        if not is_cuda_oom(error):
            raise
        del features, sources
        torch.cuda.empty_cache()
        return False


def select_largest_batch(model, args: argparse.Namespace, device: torch.device) -> int:
    lower = 0
    candidate = args.probe_initial_batch
    failed_upper: int | None = None
    while True:
        ok = try_batch_size(model, candidate, args.output_layer, device)
        print(f"Probe frame batch {candidate}: {'ok' if ok else 'OOM'}", flush=True)
        if not ok:
            failed_upper = candidate
            break
        lower = candidate
        if candidate == args.probe_ceiling_batch:
            raise RuntimeError(
                f"The configured probe ceiling ({args.probe_ceiling_batch}) also fits. "
                "Increase --probe-ceiling-batch; refusing to claim this is the largest possible batch."
            )
        candidate = min(candidate * 2, args.probe_ceiling_batch)

    assert failed_upper is not None and lower > 0
    low = lower + 1
    high = failed_upper - 1
    while low <= high:
        middle = (low + high) // 2
        ok = try_batch_size(model, middle, args.output_layer, device)
        print(f"Probe frame batch {middle}: {'ok' if ok else 'OOM'}", flush=True)
        if ok:
            lower = middle
            low = middle + 1
        else:
            high = middle - 1
    return lower


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not input_dir.is_dir():
        raise SystemExit(f"Input directory does not exist: {input_dir}")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise SystemExit("A CUDA device is required")
    torch.cuda.set_device(device)

    sources = read_manifest(args.file_list.expanduser().resolve(), input_dir)
    if args.skip_existing:
        sources = [source for source in sources if not destination_for(source, input_dir, output_dir).is_file()]
    if not sources:
        print("No files require extraction.", flush=True)
        return

    model = torch.hub.load(
        "bshall/knn-vc",
        "wavlm_large",
        trust_repo=True,
        progress=True,
        device=device,
    ).eval()
    frame_batch_size = select_largest_batch(model, args, device)
    print(
        f"Selected largest frame batch size: {frame_batch_size}; "
        f"each item is one {RECEPTIVE_FIELD}-sample WavLM window.",
        flush=True,
    )

    save_dtype = torch.float16 if args.save_dtype == "float16" else torch.float32
    source_iter = iter(sources)
    current: UtteranceWork | None = None
    completed_files = 0
    completed_frames = 0
    batch_count = 0

    while current is not None or source_iter is not None:
        segments: list[tuple[UtteranceWork, int, int]] = []
        frame_blocks: list[torch.Tensor] = []
        remaining_capacity = frame_batch_size
        completed_in_batch: list[UtteranceWork] = []

        while remaining_capacity:
            if current is None:
                try:
                    source = next(source_iter)
                except StopIteration:
                    source_iter = None
                    break
                destination = destination_for(source, input_dir, output_dir)
                windows = load_windows(source)
                if windows.shape[0] == 0:
                    save_tensor_atomically(torch.empty((0, HIDDEN_SIZE), dtype=save_dtype), destination)
                    completed_files += 1
                    continue
                current = UtteranceWork(
                    source=source,
                    destination=destination,
                    windows=windows,
                    features=torch.empty((windows.shape[0], HIDDEN_SIZE), dtype=save_dtype),
                )

            start = current.cursor
            end = min(start + remaining_capacity, current.windows.shape[0])
            frame_blocks.append(current.windows[start:end])
            segments.append((current, start, end))
            current.cursor = end
            remaining_capacity -= end - start
            if current.cursor == current.windows.shape[0]:
                completed_in_batch.append(current)
                current = None

        if not frame_blocks:
            break

        batch = torch.cat(frame_blocks, dim=0).to(device, non_blocking=False)
        with torch.inference_mode():
            embeddings, _ = model.extract_features(batch, output_layer=args.output_layer)
        embeddings = embeddings[:, 0].to(device="cpu", dtype=save_dtype)
        offset = 0
        for work, start, end in segments:
            count = end - start
            work.features[start:end].copy_(embeddings[offset:offset + count])
            offset += count
        if offset != embeddings.shape[0]:
            raise RuntimeError("Batch bookkeeping mismatch")
        del batch, embeddings

        for work in completed_in_batch:
            save_tensor_atomically(work.features, work.destination)
            completed_files += 1
            completed_frames += work.features.shape[0]
        batch_count += 1
        if batch_count % args.progress_every_batches == 0:
            print(
                f"Batches={batch_count}; files={completed_files}/{len(sources)}; "
                f"frames={completed_frames}; last_batch={offset}",
                flush=True,
            )

    print(
        f"Completed {completed_files} files and {completed_frames} frames with frame batch {frame_batch_size}.",
        flush=True,
    )


if __name__ == "__main__":
    main()
