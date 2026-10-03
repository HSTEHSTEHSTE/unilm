#!/usr/bin/env python3
"""Vocode saved WavLM frame features with KNN-VC's WavLM HiFi-GAN."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torchaudio


# Torch Hub's KNN-VC repository imports its own ``wavlm`` package. This
# checkout contains both a ``wavlm/`` directory and a local ``wavlm.py``
# module, either of which can shadow the Hub package when the script is run
# from the UniLM checkout. Exclude both local import roots and discard a
# module that may already have been resolved from them before loading Hub code.
SCRIPT_DIRECTORY = Path(__file__).resolve().parent
PROJECT_DIRECTORY = SCRIPT_DIRECTORY.parent
sys.path = [
    entry for entry in sys.path
    if Path(entry or ".").resolve() not in {SCRIPT_DIRECTORY, PROJECT_DIRECTORY}
]
sys.modules.pop("wavlm", None)


def parse_feature(value: str) -> tuple[str, Path]:
    try:
        name, path = value.split("=", maxsplit=1)
    except ValueError as error:
        raise argparse.ArgumentTypeError("--feature must have the form NAME=PATH") from error
    if not name:
        raise argparse.ArgumentTypeError("feature NAME cannot be empty")
    return name, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--feature",
        action="append",
        required=True,
        type=parse_feature,
        metavar="NAME=PATH",
        help="Saved [frames, 1024] WavLM feature tensor to vocode.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("HiFi-GAN resynthesis requires a CUDA device")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")
    hifigan, _ = torch.hub.load(
        "bshall/knn-vc",
        "hifigan_wavlm",
        trust_repo=True,
        prematched=True,
        progress=True,
        device=device,
    )
    hifigan.eval()

    for name, feature_path in args.feature:
        if not feature_path.is_file():
            raise SystemExit(f"Feature file does not exist: {feature_path}")
        features = torch.load(feature_path, map_location="cpu", weights_only=True)
        if not isinstance(features, torch.Tensor) or features.ndim != 2 or features.shape[1] != 1024:
            raise SystemExit(
                f"Expected [frames, 1024] tensor in {feature_path}; got {type(features)!r} "
                f"with shape {getattr(features, 'shape', None)}"
            )
        with torch.inference_mode():
            waveform = hifigan(features.unsqueeze(0).to(device=device, dtype=torch.float32))
        output_path = args.output_dir / f"{name}.wav"
        torchaudio.save(str(output_path), waveform.squeeze(0).detach().cpu(), 16_000)
        print(f"Wrote {output_path} from {features.shape[0]} WavLM frames", flush=True)


if __name__ == "__main__":
    main()
