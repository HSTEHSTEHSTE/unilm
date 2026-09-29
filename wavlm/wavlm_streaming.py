"""Extract WavLM features under a bounded-context streaming simulation.

Each forward pass receives a window made of prior feature frames (history),
the next feature frames to emit (step), and optional future feature frames
(look-ahead).  The script retains only the step frames from that pass.  This
recomputes overlapping windows; it is a streaming-context simulation, not a
stateful or KV-cache implementation of WavLM.
"""

import argparse
from pathlib import Path

import torch
import torchaudio
from tqdm import tqdm

from WavLM import WavLM, WavLMConfig


SAMPLE_RATE = 16000
SPEAKER_INFORMATION_LAYER = 6
DEFAULT_BATCH_SIZE = 36
DEFAULT_MAX_UTTERANCE_SECONDS = 30
CORPORA_DIR = Path('/home/hltcoe/xli/ARTS/Voice-Privacy-Challenge-2024/corpora')
CHECKPOINT_PATH = CORPORA_DIR / 'pretrained_models' / 'wavlm' / 'WavLM-Large.pt'
AUDIO_SUFFIXES = {'.flac', '.wav'}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--split',
        type=Path,
        default=Path('LibriSpeech/dev-clean'),
        help='Relative path to the input split under the corpora directory.',
    )
    parser.add_argument(
        '--step-frames',
        type=int,
        default=50,
        help=(
            'Feature frames emitted by each forward pass (default: 50 = 1 second). '
            'Set to -1 with --history-frames -1 for unbounded extraction.'
        ),
    )
    parser.add_argument(
        '--history-frames',
        type=int,
        default=100,
        help=(
            'Prior feature frames available to each forward pass (default: 100 = 2 seconds). '
            'Set to -1 to allow all prior frames; also set --step-frames -1 for unbounded extraction.'
        ),
    )
    parser.add_argument(
        '--lookahead-frames',
        type=int,
        default=0,
        help='Future feature frames available to each forward pass (default: 0).',
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help='Number of streaming windows inferred together (default: %(default)s).',
    )
    parser.add_argument(
        '--max-utterance-seconds',
        type=float,
        default=DEFAULT_MAX_UTTERANCE_SECONDS,
        help='Truncate each input to this duration before extraction (default: %(default)s).',
    )
    parser.add_argument(
        '--output-name',
        help=(
            'Name below corpora/wavlm_streaming. By default it encodes the '
            'step, history, and look-ahead settings.'
        ),
    )
    parser.add_argument(
        '--output-layer',
        type=int,
        default=SPEAKER_INFORMATION_LAYER,
        help='One-indexed WavLM layer to save (default: %(default)s).',
    )
    return parser.parse_args()


def feature_geometry(cfg):
    """Return the convolutional feature hop and receptive field in samples."""
    layers = eval(cfg.conv_feature_layers)
    hop = 1
    receptive_field = 1
    for _, kernel_size, stride in layers:
        receptive_field += (kernel_size - 1) * hop
        hop *= stride
    return hop, receptive_field


def feature_frame_count(num_samples, hop, receptive_field):
    if num_samples < receptive_field:
        return 0
    return 1 + (num_samples - receptive_field) // hop


def make_streaming_chunks(
    wav,
    total_frames,
    step_frames,
    history_frames,
    lookahead_frames,
    hop,
    receptive_field,
    normalize,
):
    """Build forward-pass windows and the local frames to retain from each."""
    chunks = []
    for emit_start in range(0, total_frames, step_frames):
        emit_end = min(emit_start + step_frames, total_frames)
        input_start = max(0, emit_start - history_frames)
        input_end = min(total_frames, emit_end + lookahead_frames)

        sample_start = input_start * hop
        sample_end = (input_end - 1) * hop + receptive_field
        chunk = wav[:, sample_start:sample_end]
        if normalize:
            # Normalizing this window (rather than the entire utterance) avoids
            # leaking samples outside the context available to this forward pass.
            chunk = torch.nn.functional.layer_norm(chunk, chunk.shape)

        chunks.append((chunk, emit_start - input_start, emit_end - emit_start))
    return chunks


def infer_chunks(model, chunks, batch_size, output_layer, device):
    """Run padded chunk batches and concatenate their retained step features."""
    output_parts = []
    for batch_start in range(0, len(chunks), batch_size):
        batch = chunks[batch_start:batch_start + batch_size]
        max_samples = max(chunk.shape[1] for chunk, _, _ in batch)
        sources = torch.zeros(len(batch), max_samples, device=device)
        padding_mask = torch.ones(
            len(batch), max_samples, dtype=torch.bool, device=device
        )

        for index, (chunk, _, _) in enumerate(batch):
            num_samples = chunk.shape[1]
            sources[index, :num_samples] = chunk[0].to(device=device)
            padding_mask[index, :num_samples] = False

        with torch.inference_mode():
            embeddings, feature_padding_mask = model.extract_features(
                sources,
                padding_mask=padding_mask,
                output_layer=output_layer,
                ret_layer_results=False,
            )

        for index, (_, keep_start, keep_frames) in enumerate(batch):
            valid_frames = int((~feature_padding_mask[index]).sum().item())
            if keep_start + keep_frames > valid_frames:
                raise RuntimeError(
                    'Streaming window produced fewer feature frames than expected: '
                    f'needed {keep_start + keep_frames}, received {valid_frames}.'
                )
            output_parts.append(
                embeddings[index, keep_start:keep_start + keep_frames].cpu()
            )

    return torch.cat(output_parts, dim=0)


def main():
    args = parse_args()
    if args.split.is_absolute():
        raise ValueError('--split must be relative to the corpora directory')
    unbounded = args.step_frames == -1
    if unbounded and args.history_frames != -1:
        raise ValueError(
            '--step-frames -1 requires --history-frames -1 for unbounded extraction'
        )
    if not unbounded and args.step_frames <= 0:
        raise ValueError('--step-frames must be positive, or -1 for unbounded extraction')
    if args.history_frames < -1:
        raise ValueError('--history-frames must be non-negative, or -1 for all prior frames')
    if args.lookahead_frames < 0:
        raise ValueError('--lookahead-frames must be non-negative')
    if args.batch_size <= 0:
        raise ValueError('--batch-size must be positive')
    if args.max_utterance_seconds <= 0:
        raise ValueError('--max-utterance-seconds must be positive')

    device = 'cuda'
    checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu')
    cfg = WavLMConfig(checkpoint['cfg'])
    model = WavLM(cfg)
    model.load_state_dict(checkpoint['model'])
    model.eval().to(device=device)

    hop, receptive_field = feature_geometry(cfg)
    frame_duration = hop / SAMPLE_RATE
    print(
        f'Feature hop: {hop} samples ({frame_duration:.3f} seconds); '
        f'receptive field: {receptive_field} samples.'
    )

    origin_dir = CORPORA_DIR / args.split
    if not origin_dir.is_dir():
        raise FileNotFoundError(f'Input split does not exist: {origin_dir}')
    wavs = sorted(
        audio_path
        for audio_path in origin_dir.rglob('*')
        if audio_path.is_file() and audio_path.suffix.lower() in AUDIO_SUFFIXES
    )

    output_name = args.output_name or (
        'unbounded'
        if unbounded
        else f'step{args.step_frames}_history{args.history_frames}_lookahead{args.lookahead_frames}'
    )
    exp_dir = CORPORA_DIR / 'wavlm_streaming' / output_name / args.split
    exp_dir.mkdir(parents=True, exist_ok=True)
    max_samples = round(args.max_utterance_seconds * SAMPLE_RATE)

    for wav_path in tqdm(wavs):
        target_path = (exp_dir / wav_path.relative_to(origin_dir)).with_suffix('.pt')
        if target_path.is_file():
            continue
        target_path.parent.mkdir(parents=True, exist_ok=True)

        wav, sample_rate = torchaudio.load(wav_path)
        if sample_rate != SAMPLE_RATE:
            wav = torchaudio.functional.resample(wav, sample_rate, SAMPLE_RATE)
        if wav.shape[0] != 1:
            raise ValueError(f'Expected mono audio, got {wav.shape[0]} channels: {wav_path}')

        wav = wav[:, :max_samples]
        total_frames = feature_frame_count(wav.shape[1], hop, receptive_field)
        if total_frames:
            step_frames = total_frames if unbounded else args.step_frames
            history_frames = total_frames if args.history_frames == -1 else args.history_frames
            chunks = make_streaming_chunks(
                wav,
                total_frames,
                step_frames,
                history_frames,
                args.lookahead_frames,
                hop,
                receptive_field,
                cfg.normalize,
            )
            features = infer_chunks(
                model,
                chunks,
                args.batch_size,
                args.output_layer,
                device,
            )
        else:
            features = torch.empty((0, cfg.encoder_embed_dim))

        temporary_path = target_path.with_name(f'{target_path.name}.tmp')
        torch.save(features.unsqueeze(0), temporary_path)
        temporary_path.replace(target_path)


if __name__ == '__main__':
    main()
