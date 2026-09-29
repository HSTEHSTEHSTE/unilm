#!/usr/bin/env python3
"""Extract unpooled WavLM frame features from a directory of audio files."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torchaudio
from tqdm import tqdm

from WavLM import WavLM, WavLMConfig


DEFAULT_CORPORA_DIR = Path('/home/hltcoe/xli/ARTS/Voice-Privacy-Challenge-2024/corpora')
DEFAULT_SPLIT = Path('LibriSpeech/train-other-360')
DEFAULT_CHECKPOINT = DEFAULT_CORPORA_DIR / 'pretrained_models/wavlm/WavLM-Large.pt'
AUDIO_SUFFIXES = {'.flac', '.wav'}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--split',
        type=Path,
        default=DEFAULT_SPLIT,
        help='Legacy relative input path below --corpora-dir (default: %(default)s).',
    )
    parser.add_argument('--corpora-dir', type=Path, default=DEFAULT_CORPORA_DIR)
    parser.add_argument('--input-dir', type=Path,
                        help='Explicit audio directory. Overrides --corpora-dir/--split.')
    parser.add_argument('--output-dir', type=Path,
                        help='Explicit output directory. Defaults to <corpora-dir>/wavlm/<split>.')
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument('--layer', type=int, default=6, help='One-indexed WavLM transformer layer.')
    parser.add_argument('--batch-size', type=int, default=36)
    parser.add_argument('--max-utterance-seconds', type=float, default=30.0)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--skip-existing', action='store_true')
    arguments = parser.parse_args()
    if arguments.batch_size < 1:
        parser.error('--batch-size must be positive')
    if arguments.layer < 1:
        parser.error('--layer must be positive')
    if arguments.max_utterance_seconds <= 0:
        parser.error('--max-utterance-seconds must be positive')
    return arguments


def main() -> None:
    arguments = parse_args()
    if arguments.input_dir is None:
        if arguments.split.is_absolute() or '..' in arguments.split.parts:
            raise SystemExit('--split must be relative to --corpora-dir')
        origin_dir = arguments.corpora_dir / arguments.split
    else:
        origin_dir = arguments.input_dir
    origin_dir = origin_dir.expanduser().resolve()
    if not origin_dir.is_dir():
        raise SystemExit(f'Input directory does not exist: {origin_dir}')

    if arguments.output_dir is None:
        output_dir = arguments.corpora_dir / 'wavlm' / arguments.split
    else:
        output_dir = arguments.output_dir
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = arguments.checkpoint.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise SystemExit(f'WavLM checkpoint does not exist: {checkpoint_path}')

    device = torch.device(arguments.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise SystemExit('CUDA was requested but is unavailable')

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    cfg = WavLMConfig(checkpoint['cfg'])
    if arguments.layer > cfg.encoder_layers:
        raise SystemExit(
            f'--layer {arguments.layer} exceeds this checkpoint\'s {cfg.encoder_layers} encoder layers'
        )
    model = WavLM(cfg)
    model.load_state_dict(checkpoint['model'])
    model.eval().to(device)

    wavs = sorted(
        audio_path
        for audio_path in origin_dir.rglob('*')
        if audio_path.is_file() and audio_path.suffix.lower() in AUDIO_SUFFIXES
    )
    if not wavs:
        raise SystemExit(f'No supported audio files found under {origin_dir}')
    print(f'Input files: {len(wavs)}', flush=True)
    print(f'Output directory: {output_dir}', flush=True)

    max_utterance_frames = round(arguments.max_utterance_seconds * 16_000)

    def target_path(audio_path: Path) -> Path:
        return (output_dir / audio_path.relative_to(origin_dir)).with_suffix('.pt')

    def save_batch(batch: list[tuple[torch.Tensor, Path]]) -> None:
        max_frames = max(waveform.shape[1] for waveform, _ in batch)
        sources = torch.zeros((len(batch), max_frames), dtype=torch.float32, device=device)
        padding_mask = torch.ones((len(batch), max_frames), dtype=torch.bool, device=device)
        for index, (waveform, _) in enumerate(batch):
            frames = waveform.shape[1]
            sources[index, :frames] = waveform[0].to(device=device, dtype=torch.float32)
            padding_mask[index, :frames] = False

        with torch.inference_mode():
            embeddings, feature_padding_mask = model.extract_features(
                sources,
                padding_mask=padding_mask,
                output_layer=arguments.layer,
                ret_layer_results=False,
            )

        if feature_padding_mask is None:
            feature_padding_mask = torch.zeros(embeddings.shape[:2], dtype=torch.bool, device=device)
        for index, (_, destination) in enumerate(batch):
            destination.parent.mkdir(parents=True, exist_ok=True)
            torch.save(embeddings[index, ~feature_padding_mask[index]].cpu(), destination)

    batch: list[tuple[torch.Tensor, Path]] = []
    for audio_path in tqdm(wavs):
        destination = target_path(audio_path)
        if arguments.skip_existing and destination.is_file():
            continue
        waveform, sample_rate = torchaudio.load(str(audio_path))
        if sample_rate != 16_000:
            waveform = torchaudio.functional.resample(waveform, sample_rate, 16_000)
        if waveform.shape[0] != 1:
            raise ValueError(f'Expected mono audio, got {waveform.shape[0]} channels: {audio_path}')
        waveform = waveform[:, :max_utterance_frames]
        if cfg.normalize:
            waveform = torch.nn.functional.layer_norm(waveform, waveform.shape)
        batch.append((waveform, destination))
        if len(batch) == arguments.batch_size:
            save_batch(batch)
            batch.clear()
    if batch:
        save_batch(batch)


if __name__ == '__main__':
    main()
