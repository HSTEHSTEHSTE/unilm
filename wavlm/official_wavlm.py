"""Loader for the sanctioned original WavLM-Large checkpoint.

Do not substitute a Hugging Face-converted WavLM checkpoint.  The approved
artifact is Microsoft's original WavLM-Large release, downloaded from the
Google Drive link in the official WavLM README.  It is byte-identical to the
checkpoint used by ``bshall/knn-vc``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import torch

from WavLM import WavLM, WavLMConfig


OFFICIAL_WAVLM_LARGE_CHECKPOINT = Path(
    "/weka/scratch/jhu/nandrew9/xli257/models/wavlm/WavLM-Large.official-release.pt"
)


def load_official_wavlm_large(
    checkpoint_path: Union[Path, str] = OFFICIAL_WAVLM_LARGE_CHECKPOINT,
    device: Optional[Union[torch.device, str]] = None,
) -> Tuple[WavLM, WavLMConfig]:
    """Load Microsoft's original, bshall-identical WavLM-Large checkpoint."""
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Official WavLM-Large checkpoint not found: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:  # Compatibility with older PyTorch environments.
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "cfg" not in checkpoint or "model" not in checkpoint:
        raise ValueError(f"{checkpoint_path} is not an original WavLM checkpoint")
    config = WavLMConfig(checkpoint["cfg"])
    model = WavLM(config)
    model.load_state_dict(checkpoint["model"], strict=True)
    if device is not None:
        model = model.to(device)
    return model.eval(), config


def feature_frame_count(num_samples: int, hop: int = 320, receptive_field: int = 400) -> int:
    """Return the number of valid WavLM frontend frames for an audio length."""
    return 0 if num_samples < receptive_field else 1 + (num_samples - receptive_field) // hop
