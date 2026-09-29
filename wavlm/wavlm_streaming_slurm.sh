#!/usr/bin/env bash

# Submit bounded-context streaming WavLM feature extraction.
#   ./wavlm_streaming_slurm.sh LibriSpeech/dev-clean
#   ./wavlm_streaming_slurm.sh LibriSpeech/test-clean 50 100 25
#   ./wavlm_streaming_slurm.sh LibriSpeech/dev-clean 50 -1 0
#   ./wavlm_streaming_slurm.sh LibriSpeech/dev-clean -1 -1 0
#   ./wavlm_streaming_slurm.sh LibriSpeech/train-other-500 5 -1 10 5-00:00:00

set -euo pipefail

usage() {
  echo "Usage: $0 [relative-split] [step-frames] [history-frames] [lookahead-frames] [time-limit]" >&2
}

if [ "$#" -lt 1 ] || [ "$#" -gt 5 ]; then
  usage
  exit 2
fi

split=$1
step_frames=${2:-50}
history_frames=${3:-100}
lookahead_frames=${4:-0}
time_limit=${5:-24:00:00}

if [[ "${split}" = /* || "${split}" == *".."* ]]; then
  echo "Split must be a relative path below the corpora directory: ${split}" >&2
  exit 2
fi
if ! [[ "${step_frames}" =~ ^-?[0-9]+$ && "${history_frames}" =~ ^-?[0-9]+$ && "${lookahead_frames}" =~ ^[0-9]+$ ]]; then
  echo "Step, history, and look-ahead frames must be integers." >&2
  exit 2
fi
if [[ "${step_frames}" == -1 && "${history_frames}" != -1 ]]; then
  echo "Step frames may be -1 only when history frames are also -1 for unbounded extraction." >&2
  exit 2
fi
if ! [[ "${time_limit}" =~ ^([0-9]+-)?[0-9]{1,2}:[0-9]{2}:[0-9]{2}$ ]]; then
  echo "Time limit must use [days-]hours:minutes:seconds format: ${time_limit}" >&2
  exit 2
fi

split_name=${split//\//_}
mode_name="step${step_frames}_history${history_frames}_lookahead${lookahead_frames}"

sbatch \
  --time="${time_limit}" \
  --job-name="wavlm_stream_${split_name}_${mode_name}" \
  --output="wavlm_stream_${split_name}_${mode_name}_%j.txt" \
  --export="ALL,WAVLM_SPLIT=${split},WAVLM_STEP_FRAMES=${step_frames},WAVLM_HISTORY_FRAMES=${history_frames},WAVLM_LOOKAHEAD_FRAMES=${lookahead_frames}" <<'EOT'
#!/usr/bin/env bash
#
#SBATCH --time=24:00:00
#SBATCH --mail-user=xli257@jhu.edu
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100

set -eo pipefail

source /home/hltcoe/xli/.bashrc
source /home/hltcoe/xli/anaconda3/etc/profile.d/conda.sh
conda activate cftts_2.8
set -u

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

root=/home/hltcoe/xli/ARTS/unilm/wavlm
corpora_dir=/home/hltcoe/xli/ARTS/Voice-Privacy-Challenge-2024/corpora
origin_dir="${corpora_dir}/${WAVLM_SPLIT}"

if [ ! -d "${origin_dir}" ]; then
  echo "Input split does not exist: ${origin_dir}" >&2
  exit 1
fi

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "Input split: ${origin_dir}"
echo "Streaming settings: step=${WAVLM_STEP_FRAMES}, history=${WAVLM_HISTORY_FRAMES}, lookahead=${WAVLM_LOOKAHEAD_FRAMES}"

cd "${root}"
python wavlm_streaming.py \
  --split "${WAVLM_SPLIT}" \
  --step-frames "${WAVLM_STEP_FRAMES}" \
  --history-frames "${WAVLM_HISTORY_FRAMES}" \
  --lookahead-frames "${WAVLM_LOOKAHEAD_FRAMES}"
EOT
