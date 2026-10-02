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
import copy
import json
import os
from pathlib import Path

import torch
import torchaudio
from tqdm import tqdm
from transformers import AutoConfig, WavLMModel, logging as transformers_logging


SAMPLE_RATE = 16_000
AUDIO_SUFFIXES = {".aac", ".flac", ".m4a", ".mp3", ".ogg", ".opus", ".wav"}
transformers_logging.set_verbosity_error()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True,
                        help="Directory containing the input audio tree.")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Directory in which to mirror feature files.")
    parser.add_argument("--model-dir", type=Path, required=True,
                        help="Local Hugging Face WavLM model directory.")
    parser.add_argument("--file-list", type=Path,
                        help="Optional relative-path manifest for one corpus shard.")
    parser.add_argument("--output-layer", type=int, default=6,
                        help="One-indexed WavLM transformer layer to save (default: %(default)s).")
    parser.add_argument("--step-frames", type=int, default=10,
                        help="Feature frames emitted by each pass (default: %(default)s).")
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
    if args.step_frames < 1:
        parser.error("--step-frames must be positive")
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


def feature_geometry(config) -> tuple[int, int]:
    hop = 1
    receptive_field = 1
    for kernel_size, stride in zip(config.conv_kernel, config.conv_stride, strict=True):
        receptive_field += (kernel_size - 1) * hop
        hop *= stride
    return hop, receptive_field


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
    total_frames: int,
    step_frames: int,
    history_frames: int,
    lookahead_frames: int,
    hop: int,
    receptive_field: int,
    normalize: bool,
) -> list[tuple[torch.Tensor, int, int]]:
    chunks = []
    for emit_start in range(0, total_frames, step_frames):
        emit_end = min(emit_start + step_frames, total_frames)
        input_start = max(0, emit_start - history_frames)
        input_end = min(total_frames, emit_end + lookahead_frames)
        sample_start = input_start * hop
        sample_end = (input_end - 1) * hop + receptive_field
        chunk = waveform[:, sample_start:sample_end]
        if normalize:
            # Window-local normalization prevents samples outside this context
            # from influencing the emitted frames.
            chunk = torch.nn.functional.layer_norm(chunk, chunk.shape)
        chunks.append((chunk, emit_start - input_start, emit_end - emit_start))
    return chunks


def infer_chunks(
    model: WavLMModel,
    chunks: list[tuple[torch.Tensor, int, int]],
    batch_size: int,
    device: torch.device,
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
        attention_mask = (~padding_mask).long()
        with torch.inference_mode():
            embeddings = model(sources, attention_mask=attention_mask).last_hidden_state
        valid_mask = model._get_feature_vector_attention_mask(embeddings.shape[1], attention_mask)
        for index, (_, keep_start, keep_frames) in enumerate(batch):
            valid_frames = int(valid_mask[index].sum().item())
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
    model_dir = args.model_dir.expanduser().resolve()
    if not input_dir.is_dir():
        raise SystemExit(f"Input directory does not exist: {input_dir}")
    if not model_dir.is_dir():
        raise SystemExit(f"WavLM model directory does not exist: {model_dir}")

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise SystemExit("This extractor requires an available CUDA device.")
    if device.index is not None and device.index >= torch.cuda.device_count():
        raise SystemExit(f"CUDA device is unavailable: {device}")
    torch.cuda.set_device(device)

    full_config = AutoConfig.from_pretrained(model_dir)
    if args.output_layer > full_config.num_hidden_layers:
        raise SystemExit(
            f"--output-layer {args.output_layer} exceeds this model's "
            f"{full_config.num_hidden_layers} encoder layers"
        )
    model_config = copy.deepcopy(full_config)
    model_config.num_hidden_layers = args.output_layer
    model = WavLMModel.from_pretrained(model_dir, config=model_config).eval().to(device)
    normalize = json.loads((model_dir / "preprocessor_config.json").read_text()).get("do_normalize", False)

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

    hop, receptive_field = feature_geometry(full_config)
    max_samples = round(args.max_utterance_seconds * SAMPLE_RATE) if args.max_utterance_seconds else None
    save_dtype = torch.float16 if args.save_dtype == "float16" else torch.float32
    print(
        f"Extracting {len(audio_paths)} files on {device}; layer={args.output_layer}; "
        f"step={args.step_frames}; history={args.history_frames}; "
        f"lookahead={args.lookahead_frames}; output: {output_dir}",
        flush=True,
    )
    for audio_path in tqdm(audio_paths, unit="file"):
        waveform = load_waveform(audio_path, max_samples)
        total_frames = feature_frame_count(waveform.shape[1], hop, receptive_field)
        if total_frames:
            chunks = make_streaming_chunks(
                waveform,
                total_frames,
                args.step_frames,
                args.history_frames,
                args.lookahead_frames,
                hop,
                receptive_field,
                normalize,
            )
            features = infer_chunks(model, chunks, args.batch_size, device)
        else:
            features = torch.empty((0, full_config.hidden_size))
        save_tensor(features.to(dtype=save_dtype), output_path(audio_path, input_dir, output_dir))


if __name__ == "__main__":
    main()
