import argparse
import torch, torchaudio
from tqdm import tqdm
from pathlib import Path
from WavLM import WavLM, WavLMConfig

SPEAKER_INFORMATION_LAYER = 6
BATCH_SIZE = 36
MAX_UTTERANCE_SECONDS = 30
CORPORA_DIR = Path('/home/hltcoe/xli/ARTS/Voice-Privacy-Challenge-2024/corpora')
AUDIO_SUFFIXES = {'.flac', '.wav'}

parser = argparse.ArgumentParser()
parser.add_argument(
    '--split',
    type=Path,
    default=Path('LibriSpeech/train-other-360'),
    help='Relative path to the input split under the corpora directory.',
)
args = parser.parse_args()
split = args.split
if split.is_absolute():
    parser.error('--split must be relative to the corpora directory')

orig_freq = 16000
transform = torchaudio.transforms.Resample(orig_freq = orig_freq, new_freq = 16000)
device = 'cuda'

# model
checkpoint_path = CORPORA_DIR / 'pretrained_models/wavlm/WavLM-Large.pt'
checkpoint = torch.load(checkpoint_path)
cfg = WavLMConfig(checkpoint['cfg'])
model = WavLM(cfg)
model.load_state_dict(checkpoint['model'])
model.eval()
model.to(device = device)

origin_dir = CORPORA_DIR / split
wavs = sorted(
    audio_path
    for audio_path in origin_dir.rglob('*')
    if audio_path.is_file() and audio_path.suffix.lower() in AUDIO_SUFFIXES
)

exp_dir = CORPORA_DIR / 'wavlm' / split
exp_dir.mkdir(parents=True, exist_ok=True)
max_utterance_frames = MAX_UTTERANCE_SECONDS * 16000


def save_batch(batch):
    """Pad a batch of utterances, infer once, and save unpadded features."""
    max_frames = max(wav.shape[1] for wav, _ in batch)
    sources = torch.zeros(len(batch), max_frames, device=device)
    padding_mask = torch.ones(len(batch), max_frames, dtype=torch.bool, device=device)

    for index, (wav, _) in enumerate(batch):
        num_frames = wav.shape[1]
        sources[index, :num_frames] = wav[0].to(device=device)
        padding_mask[index, :num_frames] = False

    with torch.inference_mode():
        embeddings, feature_padding_mask = model.extract_features(
            sources,
            padding_mask=padding_mask,
            output_layer=SPEAKER_INFORMATION_LAYER,
            ret_layer_results=False,
        )

    for index, (_, target_dir) in enumerate(batch):
        # The model padding mask is at feature-frame resolution, not sample resolution.
        valid_frames = ~feature_padding_mask[index]
        torch.save(embeddings[index, valid_frames].cpu(), target_dir)


batch = []
for wav_dir in tqdm(wavs):
    wav, fs = torchaudio.load(wav_dir)
    assert fs == orig_freq
    if fs != 16000:
        wav = transform(wav)
    if wav.shape[0] != 1:
        raise ValueError(f'Expected mono audio, got {wav.shape[0]} channels: {wav_dir}')

    wav = wav[:, :max_utterance_frames]
    if cfg.normalize:
        wav = torch.nn.functional.layer_norm(wav, wav.shape)

    target_dir = (exp_dir / wav_dir.relative_to(origin_dir)).with_suffix('.pt')
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    batch.append((wav, target_dir))

    if len(batch) == BATCH_SIZE:
        save_batch(batch)
        batch.clear()

if batch:
    save_batch(batch)
