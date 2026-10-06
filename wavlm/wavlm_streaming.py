#!/usr/bin/env python3
"""Extract bounded-context WavLM features without future look-ahead.

Each forward pass receives prior feature frames, the next step frames to emit,
and an optional number of future frames.  Only the step frames are retained.
This is a recomputing streaming-context simulation, not a stateful KV-cache
implementation.  Input and output paths are explicit so the script can be
used with a shard manifest on a single GPU.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import torchaudio
from tqdm import tqdm
from WavLM import WavLM
from official_wavlm import OFFICIAL_WAVLM_LARGE_CHECKPOINT, load_official_wavlm_large


SAMPLE_RATE = 16_000
AUDIO_SUFFIXES = {".aac", ".flac", ".m4a", ".mp3", ".ogg", ".opus", ".wav"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True,
                        help="Directory containing the input audio tree.")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Directory in which to mirror feature files.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=OFFICIAL_WAVLM_LARGE_CHECKPOINT,
        help="Microsoft original / bshall-identical WavLM-Large checkpoint.",
    )
    parser.add_argument("--file-list", type=Path,
                        help="Optional relative-path manifest for one corpus shard.")
    parser.add_argument("--output-layer", type=int, default=6,
                        help="One-indexed WavLM transformer layer to save (default: %(default)s).")
    parser.add_argument("--chunk-frames", type=int, default=10,
                        help="Hop-aligned core input duration per pass (default: %(default)s).")
    parser.add_argument("--history-frames", type=int, default=100,
                        help="Prior feature frames available to each pass (default: %(default)s).")
    parser.add_argument("--lookahead-frames", type=int, default=0,
                        help="Future feature frames available to each pass (default: %(default)s).")
    parser.add_argument("--batch-size", type=int, default=36,
                        help="Streaming windows inferred together (default: %(default)s).")
    parser.add_argument(
        "--max-utterance-seconds",
        type=float,
        default=0.0,
        help="Optional per-file truncation; 0 preserves full audio (default: %(default)s).",
    )
    parser.add_argument("--device", default="cuda:0",
                        help="Single CUDA device to use (default: %(default)s).")
    parser.add_argument("--save-dtype", choices=("float16", "float32"), default="float16",
                        help="Dtype used for saved tensors (default: %(default)s).")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Do not recompute existing feature files.")
    args = parser.parse_args()
    if args.output_layer < 1:
        parser.error("--output-layer must be positive")
    if args.chunk_frames < 2:
        parser.error("--chunk-frames must be at least two so WavLM emits one valid frame")
    if args.history_frames < 0:
        parser.error("--history-frames must be non-negative")
    if args.lookahead_frames < 0:
        parser.error("--lookahead-frames must be non-negative")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.max_utterance_seconds < 0:
        parser.error("--max-utterance-seconds cannot be negative")
    return args


def output_path(audio_path: Path, input_dir: Path, output_dir: Path) -> Path:
    return (output_dir / audio_path.relative_to(input_dir)).with_suffix(".pt")


def discover_audio(input_dir: Path) -> list[Path]:
    return sorted(
        path for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES
    )


def read_file_list(file_list: Path, input_dir: Path) -> list[Path]:
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
        if not audio_path.is_file() or audio_path.suffix.lower() not in AUDIO_SUFFIXES:
            raise SystemExit(f"Invalid audio path on line {line_number}: {audio_path}")
        if audio_path in seen:
            raise SystemExit(f"Duplicate audio path on line {line_number}: {line!r}")
        paths.append(audio_path)
        seen.add(audio_path)
    return paths


def feature_frame_count(num_samples: int, hop: int, receptive_field: int) -> int:
    if num_samples < receptive_field:
        return 0
    return 1 + (num_samples - receptive_field) // hop


def load_waveform(audio_path: Path, max_samples: int | None) -> torch.Tensor:
    waveform, sample_rate = torchaudio.load(str(audio_path))
    if waveform.numel() == 0:
        raise RuntimeError(f"Empty audio file: {audio_path}")
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sample_rate, SAMPLE_RATE)
    if max_samples is not None:
        waveform = waveform[:, :max_samples]
    return waveform.to(dtype=torch.float32)


def make_streaming_chunks(
    waveform: torch.Tensor,
    chunk_frames: int,
    history_frames: int,
    lookahead_frames: int,
    hop: int,
    receptive_field: int,
) -> list[tuple[torch.Tensor, int, int]]:
    """Build chunks with anon_baseline's USCF streaming geometry.

    A zero-lookahead core of ``N`` hop-aligned frames contains only ``N - 1``
    valid WavLM frontend outputs: the last would require the additional
    80-sample receptive-field tail.  anon_baseline therefore advances by
    ``N - 1`` frames in that case.  With look-ahead, include that tail and
    advance by the requested core size.
    """
    chunks = []
    chunk_samples = chunk_frames * hop
    history_samples = history_frames * hop
    lookahead_samples = lookahead_frames * hop
    emission_samples = (
        chunk_samples + lookahead_samples
        if lookahead_samples
        else chunk_samples - hop
    )
    if emission_samples <= 0:
        raise ValueError("chunk duration is too short for WavLM's valid convolution")

    waveform_length = waveform.shape[1]
    for current_start in range(0, waveform_length, emission_samples):
        core_end = min(current_start + chunk_samples, waveform_length)
        sample_start = max(0, current_start - history_samples)
        # This is get_uscf_streaming_chunk_bounds() from anon_baseline:
        # only explicit look-ahead receives the 80-sample frontend tail.
        sample_end = min(
            core_end + lookahead_samples + (receptive_field - hop if lookahead_samples else 0),
            waveform_length,
        )
        chunk = waveform[:, sample_start:sample_end]
        keep_start = (current_start - sample_start) // hop
        valid_frames = feature_frame_count(chunk.shape[1], hop, receptive_field)
        keep_frames = valid_frames - keep_start if core_end >= waveform_length else emission_samples // hop
        if keep_frames < 0:
            raise RuntimeError("Streaming chunk has a negative emitted frame count")
        chunks.append((chunk, keep_start, keep_frames))
        # The final core can end before the next hop-aligned source position.
        # anon_baseline terminates at this point; continuing would construct a
        # phantom tail chunk with no valid frame to emit.
        if core_end >= waveform_length:
            break
    return chunks


def infer_chunks(
    model: WavLM,
    chunks: list[tuple[torch.Tensor, int, int]],
    batch_size: int,
    device: torch.device,
    output_layer: int,
    hop: int,
    receptive_field: int,
) -> torch.Tensor:
    output_parts = []
    for batch_start in range(0, len(chunks), batch_size):
        batch = chunks[batch_start:batch_start + batch_size]
        max_samples = max(chunk.shape[1] for chunk, _, _ in batch)
        sources = torch.zeros((len(batch), max_samples), dtype=torch.float32, device=device)
        padding_mask = torch.ones((len(batch), max_samples), dtype=torch.bool, device=device)
        for index, (chunk, _, _) in enumerate(batch):
            samples = chunk.shape[1]
            sources[index, :samples] = chunk[0].to(device)
            padding_mask[index, :samples] = False
        with torch.inference_mode():
            embeddings, _ = model.extract_features(
                sources, padding_mask=padding_mask, output_layer=output_layer
            )
        for index, (chunk, keep_start, keep_frames) in enumerate(batch):
            valid_frames = feature_frame_count(chunk.shape[1], hop, receptive_field)
            if keep_start + keep_frames > valid_frames:
                raise RuntimeError(
                    "Streaming window produced fewer feature frames than expected: "
                    f"needed {keep_start + keep_frames}, received {valid_frames}."
                )
            output_parts.append(embeddings[index, keep_start:keep_start + keep_frames].cpu())
    return torch.cat(output_parts, dim=0)


def save_tensor(tensor: torch.Tensor, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        torch.save(tensor, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    if not input_dir.is_dir():
        raise SystemExit(f"Input directory does not exist: {input_dir}")
    if not checkpoint_path.is_file():
        raise SystemExit(f"Official WavLM checkpoint does not exist: {checkpoint_path}")

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise SystemExit("This extractor requires an available CUDA device.")
    if device.index is not None and device.index >= torch.cuda.device_count():
        raise SystemExit(f"CUDA device is unavailable: {device}")
    torch.cuda.set_device(device)

    model, full_config = load_official_wavlm_large(checkpoint_path, device)
    if args.output_layer > full_config.encoder_layers:
        raise SystemExit(
            f"--output-layer {args.output_layer} exceeds this model's "
            f"{full_config.encoder_layers} encoder layers"
        )

    audio_paths = (
        read_file_list(args.file_list.expanduser().resolve(), input_dir)
        if args.file_list is not None
        else discover_audio(input_dir)
    )
    if args.skip_existing:
        audio_paths = [
            path for path in audio_paths
            if not output_path(path, input_dir, output_dir).is_file()
        ]
    if not audio_paths:
        print("No files require extraction.", flush=True)
        return

    hop, receptive_field = 320, 400
    max_samples = round(args.max_utterance_seconds * SAMPLE_RATE) if args.max_utterance_seconds else None
    save_dtype = torch.float16 if args.save_dtype == "float16" else torch.float32
    print(
        f"Extracting {len(audio_paths)} files on {device}; layer={args.output_layer}; "
        f"chunk={args.chunk_frames}; history={args.history_frames}; lookahead={args.lookahead_frames}; "
        f"anon-baseline emission stride="
        f"{args.chunk_frames + args.lookahead_frames if args.lookahead_frames else args.chunk_frames - 1}; "
        f"checkpoint={checkpoint_path}; output: {output_dir}",
        flush=True,
    )
    for audio_path in tqdm(audio_paths, unit="file"):
        waveform = load_waveform(audio_path, max_samples)
        if feature_frame_count(waveform.shape[1], hop, receptive_field):
            chunks = make_streaming_chunks(
                waveform,
                args.chunk_frames,
                args.history_frames,
                args.lookahead_frames,
                hop,
                receptive_field,
            )
            features = infer_chunks(
                model, chunks, args.batch_size, device, args.output_layer, hop, receptive_field
            )
        else:
            features = torch.empty((0, full_config.encoder_embed_dim))
        save_tensor(features.to(dtype=save_dtype), output_path(audio_path, input_dir, output_dir))


if __name__ == "__main__":
    main()
