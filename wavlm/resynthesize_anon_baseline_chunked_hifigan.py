#!/usr/bin/env python3
"""Reproduce anon_baseline's corrected chunked WavLM -> KNN-VC HiFi-GAN path."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torchaudio
import torchaudio.functional as audio_functional


# Torch Hub's KNN-VC repository imports its bundled ``wavlm`` package. Avoid
# resolving this checkout's local module instead.
SCRIPT_DIRECTORY = Path(__file__).resolve().parent
PROJECT_DIRECTORY = SCRIPT_DIRECTORY.parent
sys.path = [
    entry for entry in sys.path
    if Path(entry or ".").resolve() not in {SCRIPT_DIRECTORY, PROJECT_DIRECTORY}
]
sys.modules.pop("wavlm", None)

SAMPLE_RATE = 16_000
FEATURE_HOP = 320
WAVLM_RECEPTIVE_FIELD = 400


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-seconds", type=float, default=0.2)
    parser.add_argument("--memory-seconds", type=float, default=60.0)
    parser.add_argument(
        "--emit-frames",
        type=int,
        help="Emit this many valid WavLM frames per chunk instead of using --chunk-seconds.",
    )
    parser.add_argument(
        "--history-frames",
        type=int,
        help="Available preceding WavLM frames when --emit-frames is used.",
    )
    return parser.parse_args()


def to_hop_samples(seconds: float) -> int:
    return round(seconds * SAMPLE_RATE / FEATURE_HOP) * FEATURE_HOP


def main() -> None:
    args = parse_args()
    if not args.audio.is_file():
        raise SystemExit(f"Audio does not exist: {args.audio}")
    if args.emit_frames is None:
        if args.history_frames is not None:
            raise SystemExit("--history-frames requires --emit-frames")
        if args.chunk_seconds <= 0 or args.memory_seconds < 0:
            raise SystemExit("chunk-seconds must be positive and memory-seconds non-negative")
        chunk_length = to_hop_samples(args.chunk_seconds)
        chunk_memory = to_hop_samples(args.memory_seconds)
        if chunk_length <= FEATURE_HOP:
            raise SystemExit("chunk-seconds must be longer than one 20 ms WavLM feature hop")
        # Match anon_baseline.get_wavlm_emission_step() for no look-ahead: the
        # valid-convolution frontend produces one fewer frame than nominal input
        # hops, so a 200 ms / 10-hop input emits nine 20-ms feature frames.
        emission_step = min(chunk_length, chunk_length - FEATURE_HOP)
    else:
        if args.emit_frames < 1 or args.history_frames is None or args.history_frames < 0:
            raise SystemExit("--emit-frames must be positive and --history-frames must be non-negative")
        # A valid-convolution WavLM frontend needs one extra 20-ms hop of input
        # to yield the requested number of frames: one emitted frame therefore
        # uses a two-hop (640-sample) input chunk, exactly as anon_baseline's
        # chunk geometry does for its larger chunks.
        chunk_length = (args.emit_frames + 1) * FEATURE_HOP
        emission_step = args.emit_frames * FEATURE_HOP
        chunk_memory = args.history_frames * FEATURE_HOP

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    device = torch.device("cuda:0")
    waveform, sample_rate = torchaudio.load(str(args.audio))
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != SAMPLE_RATE:
        waveform = audio_functional.resample(waveform, sample_rate, SAMPLE_RATE)
    waveform = waveform.to(device)

    wavlm = torch.hub.load("bshall/knn-vc", "wavlm_large", trust_repo=True, progress=True, device=device)
    hifigan, _ = torch.hub.load(
        "bshall/knn-vc", "hifigan_wavlm", trust_repo=True, prematched=True, progress=True, device=device
    )
    wavlm.eval()
    hifigan.eval()

    output_chunks: list[torch.Tensor] = []
    current_start = 0
    with torch.inference_mode():
        while current_start < waveform.shape[1]:
            core_end = min(current_start + chunk_length, waveform.shape[1])
            chunk_start = max(current_start - chunk_memory, 0)
            current_chunk = waveform[:, chunk_start:core_end]
            if current_chunk.shape[1] < WAVLM_RECEPTIVE_FIELD:
                break
            features, _ = wavlm.extract_features(current_chunk, output_layer=6)
            decoded = hifigan(features).squeeze(0)
            if decoded.ndim == 1:
                decoded = decoded.unsqueeze(0)
            decoded = decoded[:, : features.shape[1] * FEATURE_HOP]
            emit_start = ((current_start - chunk_start) // FEATURE_HOP) * FEATURE_HOP
            is_final = core_end >= waveform.shape[1]
            if is_final:
                output_chunks.append(decoded[:, emit_start:].cpu())
                break
            output_chunks.append(decoded[:, emit_start:emit_start + emission_step].cpu())
            current_start += emission_step

    if not output_chunks:
        raise SystemExit("No audio was emitted")
    output = torch.cat(output_chunks, dim=1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(args.output), output, SAMPLE_RATE)
    print(
        f"Wrote {args.output}; input={chunk_length // FEATURE_HOP} nominal frames; "
        f"emission={emission_step // FEATURE_HOP} frames; history={chunk_memory // FEATURE_HOP} frames; "
        f"samples={output.shape[1]}",
        flush=True,
    )


if __name__ == "__main__":
    main()
