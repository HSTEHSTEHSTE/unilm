#!/usr/bin/env python3
"""Extract batched WavLM frame features from a recursively nested audio tree.

Each input ``relative/path/audio.flac`` is written as
``<output-dir>/relative/path/audio.flac.pt``.  The saved value is a CPU tensor
with shape ``(frames, hidden_size)``; no temporal pooling is applied.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
from typing import Iterable

import torch
import torchaudio
from tqdm import tqdm
from transformers import AutoConfig, WavLMModel, logging as transformers_logging


AUDIO_SUFFIXES = {".aac", ".flac", ".m4a", ".mp3", ".ogg", ".opus", ".wav"}
TARGET_SAMPLE_RATE = 16_000
transformers_logging.set_verbosity_error()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True,
                        help="Directory to search recursively for audio files.")
    parser.add_argument(
        "--file-list",
        type=Path,
        help=("Optional newline-delimited list of audio paths relative to --input-dir. "
              "Use this to process one shard of a corpus."),
    )
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Directory in which to mirror feature files.")
    parser.add_argument("--model-dir", type=Path, required=True,
                        help="Local Hugging Face WavLM model directory.")
    parser.add_argument("--layer", type=int, default=6,
                        help="One-indexed transformer layer to save (default: %(default)s).")
    parser.add_argument("--device", default="cuda:0",
                        help="Single CUDA device to use (default: %(default)s).")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help=("Upper bound for the startup batch-size probe and dynamic batches "
              "(default: %(default)s)."),
    )
    parser.add_argument(
        "--max-batch-seconds",
        type=float,
        default=30.0,
        help=("Also cap padded audio per batch, measured as batch_size × longest duration. "
              "Set to 0 to disable (default: %(default)s)."),
    )
    parser.add_argument("--save-dtype", choices=("float16", "float32"), default="float32",
                        help="Dtype used for saved tensors (default: %(default)s).")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Do not recompute feature files already present in --output-dir.")
    args = parser.parse_args()

    if args.layer < 1:
        parser.error("--layer must be positive")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.max_batch_seconds < 0:
        parser.error("--max-batch-seconds cannot be negative")
    return args


def output_path(audio_path: Path, input_dir: Path, output_dir: Path) -> Path:
    """Keep the full audio filename to avoid .flac/.wav stem collisions."""
    relative_path = audio_path.relative_to(input_dir)
    return output_dir / relative_path.parent / f"{relative_path.name}.pt"


def discover_audio(input_dir: Path) -> list[Path]:
    return sorted(
        path for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES
    )


def read_file_list(file_list: Path, input_dir: Path) -> list[Path]:
    """Read a relative-path manifest without permitting paths outside the input tree."""
    if not file_list.is_file():
        raise SystemExit(f"File list does not exist: {file_list}")

    paths: list[Path] = []
    seen: set[Path] = set()
    for line_number, line in enumerate(file_list.read_text().splitlines(), start=1):
        relative_path = Path(line.strip())
        if not line.strip():
            continue
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise SystemExit(f"Invalid relative path on line {line_number}: {line!r}")
        audio_path = (input_dir / relative_path).resolve()
        try:
            audio_path.relative_to(input_dir)
        except ValueError as error:
            raise SystemExit(f"Path on line {line_number} escapes --input-dir: {line!r}") from error
        if not audio_path.is_file():
            raise SystemExit(f"Audio file on line {line_number} does not exist: {audio_path}")
        if audio_path.suffix.lower() not in AUDIO_SUFFIXES:
            raise SystemExit(f"Unsupported audio type on line {line_number}: {audio_path}")
        if audio_path in seen:
            raise SystemExit(f"Duplicate audio path on line {line_number}: {line!r}")
        paths.append(audio_path)
        seen.add(audio_path)
    return paths


def estimated_samples(audio_path: Path) -> int:
    """Return the approximate 16 kHz length used to build efficient batches."""
    info = torchaudio.info(str(audio_path))
    if info.sample_rate <= 0 or info.num_frames < 0:
        raise RuntimeError(f"Could not determine duration for {audio_path}")
    return round(info.num_frames * TARGET_SAMPLE_RATE / info.sample_rate)


def make_batches(
    pending: Iterable[tuple[int, Path]],
    worst_case_samples: int,
    worst_case_batch_size: int,
    max_batch_size: int,
    max_batch_seconds: float,
) -> Iterable[list[Path]]:
    """Length-sort and scale capacity linearly from the measured worst case."""
    max_padded_samples = round(max_batch_seconds * TARGET_SAMPLE_RATE)
    batch: list[Path] = []
    longest = 0
    for samples, audio_path in pending:
        candidate_longest = max(longest, samples)
        dynamic_batch_size = min(
            max_batch_size,
            max(1, worst_case_batch_size * worst_case_samples // max(candidate_longest, 1)),
        )
        would_exceed_duration = (
            max_padded_samples > 0
            and batch
            and (len(batch) + 1) * candidate_longest > max_padded_samples
        )
        if len(batch) >= dynamic_batch_size or would_exceed_duration:
            yield batch
            batch = []
            longest = 0
        batch.append(audio_path)
        longest = max(longest, samples)
    if batch:
        yield batch


def load_waveform(audio_path: Path, normalize: bool) -> torch.Tensor:
    waveform, sample_rate = torchaudio.load(str(audio_path))
    if waveform.numel() == 0:
        raise RuntimeError(f"Empty audio file: {audio_path}")
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != TARGET_SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sample_rate, TARGET_SAMPLE_RATE)
    waveform = waveform.to(dtype=torch.float32)
    if normalize:
        waveform = torch.nn.functional.layer_norm(waveform, waveform.shape)
    return waveform


def save_tensor(tensor: torch.Tensor, destination: Path) -> None:
    """Atomically publish a completed feature file, so interrupted jobs can resume."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        torch.save(tensor, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def is_cuda_oom(error: RuntimeError) -> bool:
    return isinstance(error, torch.cuda.OutOfMemoryError) or "out of memory" in str(error).lower()


def probe_batch_size(
    model: WavLMModel,
    waveform: torch.Tensor,
    device: torch.device,
    upper_bound: int,
) -> int:
    """Find the largest batch of the longest input that fits on this GPU."""
    def fits(batch_size: int) -> bool:
        sources = waveform.to(device).repeat(batch_size, 1)
        padding_mask = torch.zeros(sources.shape, dtype=torch.bool, device=device)
        try:
            with torch.inference_mode():
                model(sources, attention_mask=(~padding_mask).long())
            torch.cuda.synchronize(device)
            return True
        except RuntimeError as error:
            if not is_cuda_oom(error):
                raise
            return False
        finally:
            del sources, padding_mask
            torch.cuda.empty_cache()

    if not fits(1):
        raise RuntimeError(
            "The longest audio file does not fit on the selected GPU even at batch size 1. "
            "Use a shorter input or a GPU with more memory."
        )

    largest_fit = 1
    first_failure = upper_bound + 1
    candidate = 2
    while candidate <= upper_bound:
        if fits(candidate):
            largest_fit = candidate
            candidate *= 2
        else:
            first_failure = candidate
            break
    if largest_fit == upper_bound or candidate > upper_bound:
        if largest_fit < upper_bound and fits(upper_bound):
            return upper_bound
        first_failure = min(first_failure, upper_bound)

    low, high = largest_fit + 1, first_failure - 1
    while low <= high:
        candidate = (low + high) // 2
        if fits(candidate):
            largest_fit = candidate
            low = candidate + 1
        else:
            high = candidate - 1
    return largest_fit


def extract_batch(
    model: WavLMModel,
    paths: list[Path],
    normalize: bool,
    device: torch.device,
    input_dir: Path,
    output_dir: Path,
    save_dtype: torch.dtype,
) -> None:
    waveforms = [load_waveform(path, normalize) for path in paths]
    max_samples = max(waveform.shape[1] for waveform in waveforms)
    sources = torch.zeros((len(waveforms), max_samples), dtype=torch.float32, device=device)
    padding_mask = torch.ones((len(waveforms), max_samples), dtype=torch.bool, device=device)
    for index, waveform in enumerate(waveforms):
        samples = waveform.shape[1]
        sources[index, :samples] = waveform[0].to(device)
        padding_mask[index, :samples] = False

    with torch.inference_mode():
        embeddings = model(sources, attention_mask=(~padding_mask).long()).last_hidden_state
    feature_padding_mask = ~model._get_feature_vector_attention_mask(
        embeddings.shape[1], (~padding_mask).long()
    )
    for index, audio_path in enumerate(paths):
        features = embeddings[index, ~feature_padding_mask[index]].to("cpu", dtype=save_dtype)
        save_tensor(features, output_path(audio_path, input_dir, output_dir))


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    model_dir = args.model_dir.expanduser().resolve()

    if not input_dir.is_dir():
        raise SystemExit(f"Input directory does not exist: {input_dir}")
    if not model_dir.is_dir():
        raise SystemExit(f"WavLM model directory does not exist: {model_dir}")

    device = torch.device(args.device)
    if device.type != "cuda":
        raise SystemExit("This extractor is single-GPU only; --device must name a CUDA device.")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")
    if device.index is not None and device.index >= torch.cuda.device_count():
        raise SystemExit(f"CUDA device is unavailable: {device}")
    torch.cuda.set_device(device)

    full_config = AutoConfig.from_pretrained(model_dir)
    if args.layer > full_config.num_hidden_layers:
        raise SystemExit(
            f"--layer {args.layer} exceeds this model's {full_config.num_hidden_layers} encoder layers"
        )
    model_config = copy.deepcopy(full_config)
    model_config.num_hidden_layers = args.layer
    model = WavLMModel.from_pretrained(model_dir, config=model_config).eval().to(device)
    preprocessor_path = model_dir / "preprocessor_config.json"
    normalize = json.loads(preprocessor_path.read_text()).get("do_normalize", False)

    audio_paths = (
        read_file_list(args.file_list.expanduser().resolve(), input_dir)
        if args.file_list is not None
        else discover_audio(input_dir)
    )
    if not audio_paths:
        raise SystemExit(f"No supported audio files found under {input_dir}")
    if args.skip_existing:
        audio_paths = [
            path for path in audio_paths
            if not output_path(path, input_dir, output_dir).is_file()
        ]
    if not audio_paths:
        print("No files require extraction.", flush=True)
        return

    pending = sorted(
        ((estimated_samples(path), path) for path in audio_paths),
        key=lambda item: item[0],
    )
    worst_case_samples, worst_case_path = pending[-1]
    worst_case_waveform = load_waveform(worst_case_path, normalize)
    worst_case_batch_size = probe_batch_size(
        model, worst_case_waveform, device, args.batch_size
    )
    del worst_case_waveform
    torch.cuda.empty_cache()

    save_dtype = torch.float16 if args.save_dtype == "float16" else torch.float32
    print(
        f"Extracting {len(audio_paths)} files on {device}; output: {output_dir}\n"
        f"Startup probe: {worst_case_path} ({worst_case_samples / TARGET_SAMPLE_RATE:.1f}s) "
        f"fits batch size {worst_case_batch_size}. "
        "Shorter files scale linearly up to --batch-size.",
        flush=True,
    )
    progress = tqdm(total=len(audio_paths), unit="file")

    def extract_with_oom_retry(paths: list[Path]) -> None:
        try:
            extract_batch(
                model, paths, normalize, device, input_dir, output_dir, save_dtype
            )
        except RuntimeError as error:
            if not is_cuda_oom(error):
                raise
            torch.cuda.empty_cache()
            if len(paths) == 1:
                raise RuntimeError(f"CUDA out of memory for {paths[0]}") from error
            midpoint = len(paths) // 2
            print(f"CUDA OOM; retrying {len(paths)} files as two smaller batches.", flush=True)
            extract_with_oom_retry(paths[:midpoint])
            extract_with_oom_retry(paths[midpoint:])
            return
        progress.update(len(paths))

    for paths in make_batches(
        pending,
        worst_case_samples,
        worst_case_batch_size,
        args.batch_size,
        args.max_batch_seconds,
    ):
        extract_with_oom_retry(paths)
    progress.close()


if __name__ == "__main__":
    main()
