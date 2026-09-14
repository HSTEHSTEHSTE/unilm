#!/usr/bin/env bash

# Submit WavLM feature extraction. Defaults to LibriSpeech/dev-clean.
#   ./wavlm_slurm.sh
#   ./wavlm_slurm.sh LibriSpeech/test-clean

set -euo pipefail

if [ "$#" -gt 1 ]; then
  echo "Usage: $0 [relative-split]" >&2
  exit 2
fi

split=${1:-LibriSpeech/train-clean-100}
if [[ "${split}" = /* || "${split}" == *".."* ]]; then
  echo "Split must be a relative path below the corpora directory: ${split}" >&2
  exit 2
fi

split_name=${split//\//_}

sbatch \
  --job-name="wavlm_${split_name}" \
  --output="wavlm_${split_name}_%j.txt" \
  --export="ALL,WAVLM_SPLIT=${split}" <<'EOT'
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

cd "${root}"
python wavlm.py --split "${WAVLM_SPLIT}"
EOT
