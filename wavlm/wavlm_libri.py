import os
import torch, torchaudio
from tqdm import tqdm
from pathlib import Path
from WavLM import WavLM, WavLMConfig

SPEAKER_INFORMATION_LAYER = 6

suffix = 'B3'
feature_name = 'wavlm_sample'
splits = [
    # {
    #     'name': 'train-clean-360/',
    #     'origin_dir': '/home/hltcoe/xli/ARTS/Voice-Privacy-Challenge-2024/corpora/LibriSpeech/',
    #     'wav_extension': 'flac',
    #     # 'exp_dir': '/home/hltcoe/xli/ARTS/Voice-Privacy-Challenge-2024/corpora/exp/corpora/LibriSpeech_smooth/' + feature_name + '/',
    #     'exp_dir': '/home/hltcoe/xli/ARTS/Voice-Privacy-Challenge-2024/corpora/exp/corpora/LibriSpeech/' + feature_name + '/',
    # },
    {
        'name': 'libri_dev_enrolls_' + suffix + '/',
        'origin_dir': '/home/hltcoe/xli/ARTS/Voice-Privacy-Challenge-2024/data/',
        'wav_extension': 'wav',
        'exp_dir': '/home/hltcoe/xli/ARTS/Voice-Privacy-Challenge-2024/corpora/exp/' + suffix + '/' + feature_name + '/',
    },
    {
        'name': 'libri_dev_trials_f_' + suffix + '/',
        'origin_dir': '/home/hltcoe/xli/ARTS/Voice-Privacy-Challenge-2024/data/',
        'wav_extension': 'wav',
        'exp_dir': '/home/hltcoe/xli/ARTS/Voice-Privacy-Challenge-2024/corpora/exp/' + suffix + '/' + feature_name + '/',
    },
    {
        'name': 'libri_dev_trials_m_' + suffix + '/',
        'origin_dir': '/home/hltcoe/xli/ARTS/Voice-Privacy-Challenge-2024/data/',
        'wav_extension': 'wav',
        'exp_dir': '/home/hltcoe/xli/ARTS/Voice-Privacy-Challenge-2024/corpora/exp/' + suffix + '/' + feature_name + '/',
    },
    {
        'name': 'libri_test_enrolls_' + suffix + '/',
        'origin_dir': '/home/hltcoe/xli/ARTS/Voice-Privacy-Challenge-2024/data/',
        'wav_extension': 'wav',
        'exp_dir': '/home/hltcoe/xli/ARTS/Voice-Privacy-Challenge-2024/corpora/exp/' + suffix + '/' + feature_name + '/',
    },
    {
        'name': 'libri_test_trials_f_' + suffix + '/',
        'origin_dir': '/home/hltcoe/xli/ARTS/Voice-Privacy-Challenge-2024/data/',
        'wav_extension': 'wav',
        'exp_dir': '/home/hltcoe/xli/ARTS/Voice-Privacy-Challenge-2024/corpora/exp/' + suffix + '/' + feature_name + '/',
    },
    {
        'name': 'libri_test_trials_m_' + suffix + '/',
        'origin_dir': '/home/hltcoe/xli/ARTS/Voice-Privacy-Challenge-2024/data/',
        'wav_extension': 'wav',
        'exp_dir': '/home/hltcoe/xli/ARTS/Voice-Privacy-Challenge-2024/corpora/exp/' + suffix + '/' + feature_name + '/',
    },
    {
        'name': 'train-clean-360_' + suffix + '/',
        'origin_dir': '/home/hltcoe/xli/ARTS/Voice-Privacy-Challenge-2024/data/',
        'wav_extension': 'wav',
        'exp_dir': '/home/hltcoe/xli/ARTS/Voice-Privacy-Challenge-2024/corpora/exp/' + suffix + '/' + feature_name + '/',
    },
    # {
    #     'name': 'libri_dev/',
    #     'origin_dir': '/home/hltcoe/xli/ARTS/Voice-Privacy-Challenge-2024/data/',
    #     'wav_extension': 'wav',
    #     'exp_dir': '/home/hltcoe/xli/ARTS/Voice-Privacy-Challenge-2024/corpora/exp/corpora/LibriSpeech/' + feature_name + '/',
    # },
    # {
    #     'name': 'libri_test/',
    #     'origin_dir': '/home/hltcoe/xli/ARTS/Voice-Privacy-Challenge-2024/data/',
    #     'wav_extension': 'wav',
    #     'exp_dir': '/home/hltcoe/xli/ARTS/Voice-Privacy-Challenge-2024/corpora/exp/corpora/LibriSpeech/' + feature_name + '/',
    # }
]


for split_dict in splits:
    split = split_dict['name']
    transform = torchaudio.transforms.Resample(orig_freq = 16000, new_freq = 16000)

    origin_dir = split_dict['origin_dir'] + split
    wavs = [os.path.join(dp, f) for dp, dn, filenames in os.walk(origin_dir) for f in filenames if os.path.splitext(f)[-1] == '.' + split_dict['wav_extension']]

    exp_dir = split_dict['exp_dir'] + split


    # model
    device = 'cuda'
    checkpoint = torch.load('/home/hltcoe/xli/ARTS/Voice-Privacy-Challenge-2024/corpora/pretrained_models/wavlm/WavLM-Large.pt')
    cfg = WavLMConfig(checkpoint['cfg'])
    model = WavLM(cfg)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    model.to(device = device)


    for wav_id, wav_dir in enumerate(tqdm(wavs)):
        target_dir = exp_dir + wav_dir[len(origin_dir):-(len(split_dict['wav_extension']) + 1)]
        target_dir += '.pt'
        # if not os.path.isfile(target_dir):
        wav, fs = torchaudio.load(wav_dir)
        wav = transform(wav)
        if wav.shape[1] > 960000 * 3: # 3 min
            wav = wav[:, :960000 * 3]
        wav = wav.to(device = device)
        embeddings = model.extract_features(wav, output_layer = SPEAKER_INFORMATION_LAYER, ret_layer_results = False)[0]
        Path(target_dir).parent.mkdir(parents = True, exist_ok = True)
        breakpoint()

        # # pool embeddings over time
        # embeddings = torch.mean(embeddings, dim = 1, keepdim = True)
        
        # sample from embedding
        sample_length = 10
        frame_choice = torch.randint(low = 0, high = embeddings.shape[1], size = [sample_length])
        embeddings = embeddings[:, frame_choice, :]
        embeddings = torch.sum(embeddings, dim = 1, keepdim = True)
        embeddings = torch.div(embeddings, sample_length)

        torch.save(embeddings, target_dir)