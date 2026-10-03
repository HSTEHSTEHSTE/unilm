#!/usr/bin/env python3
"""Compare native KNN-VC features with an existing streamed WavLM tensor."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torchaudio
import torchaudio.functional as audio_functional


# KNN-VC's Torch Hub repository imports its bundled ``wavlm`` package. Avoid
# shadowing it with this repository's top-level ``wavlm.py`` module.
SCRIPT_DIRECTORY = Path(__file__).resolve().parent
sys.path = [entry for entry in sys.path if Path(entry or ".").resolve() != SCRIPT_DIRECTORY]


def load_audio(path: Path, device: torch.device) -> torch.Tensor:
    waveform, sample_rate = torchaudio.load(str(path))
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != 16_000:
        waveform = audio_functional.resample(waveform, sample_rate, 16_000)
    return waveform.to(device)


def save_vocoded(hifigan, features: torch.Tensor, path: Path, device: torch.device) -> torch.Tensor:
    with torch.inference_mode():
        waveform = hifigan(features.unsqueeze(0).to(device=device, dtype=torch.float32))
    waveform = waveform.squeeze(0).detach().cpu()
    torchaudio.save(str(path), waveform, 16_000)
    return waveform


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", required=True, type=Path)
    parser.add_argument("--streamed-features", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("This diagnostic requires CUDA")
    if not args.audio.is_file() or not args.streamed_features.is_file():
        raise SystemExit("The audio file or streamed feature tensor does not exist")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")
    wavlm = torch.hub.load("bshall/knn-vc", "wavlm_large", trust_repo=True, progress=True, device=device)
    hifigan, _ = torch.hub.load(
        "bshall/knn-vc", "hifigan_wavlm", trust_repo=True, prematched=True, progress=True, device=device
    )

    waveform = load_audio(args.audio, device)
    with torch.inference_mode():
        native_features, _ = wavlm.extract_features(waveform, output_layer=6)
    native_features = native_features.squeeze(0).detach().cpu()

    streamed_features = torch.load(args.streamed_features, map_location="cpu", weights_only=True)
    if streamed_features.ndim == 3 and streamed_features.shape[0] == 1:
        streamed_features = streamed_features.squeeze(0)
    if streamed_features.ndim != 2 or streamed_features.shape[1] != native_features.shape[1]:
        raise SystemExit(
            f"Unexpected streamed feature shape {tuple(streamed_features.shape)}; "
            f"native shape is {tuple(native_features.shape)}"
        )
    if streamed_features.shape != native_features.shape:
        raise SystemExit(
            f"Feature-frame count differs: streamed {tuple(streamed_features.shape)}, "
            f"native {tuple(native_features.shape)}"
        )

    streamed_features = streamed_features.float()
    metrics = {
        "feature_shape": list(native_features.shape),
        "feature_cosine": float(torch.nn.functional.cosine_similarity(
            native_features.flatten(), streamed_features.flatten(), dim=0
        )),
        "feature_mae": float((native_features - streamed_features).abs().mean()),
        "feature_rmse": float((native_features - streamed_features).square().mean().sqrt()),
    }
    native_audio = save_vocoded(hifigan, native_features, args.output_dir / "native_knnvc_full.wav", device)
    streamed_audio = save_vocoded(hifigan, streamed_features, args.output_dir / "streamed_step10_history100.wav", device)
    metrics["waveform_shape"] = list(native_audio.shape)
    metrics["waveform_cosine"] = float(torch.nn.functional.cosine_similarity(
        native_audio.flatten(), streamed_audio.flatten(), dim=0
    ))
    metrics["waveform_mae"] = float((native_audio - streamed_audio).abs().mean())
    metrics["waveform_rmse"] = float((native_audio - streamed_audio).square().mean().sqrt())
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
