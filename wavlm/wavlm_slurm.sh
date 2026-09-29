#!/usr/bin/env bash

# Submit scratch-staged WavLM feature extraction on a V100.
#
# Legacy usage:
#   ./wavlm_slurm.sh LibriSpeech/train-clean-100
#
# For an explicit corpus/output/model, set WAVLM_INPUT_DIR, WAVLM_OUTPUT_DIR,
# and WAVLM_CHECKPOINT before invoking this launcher.

set -euo pipefail

if [ "$#" -gt 1 ]; then
  echo "Usage: $0 [relative-split]" >&2
  exit 2
fi

split=${1:-LibriSpeech/train-clean-100}
if [[ "${split}" = /* || "${split}" == *".."* ]]; then
  echo "Split must be a relative path below the legacy corpora directory: ${split}" >&2
  exit 2
fi

root=/home/hltcoe/xli/ARTS/unilm/wavlm
legacy_corpora=/home/hltcoe/xli/ARTS/Voice-Privacy-Challenge-2024/corpora
input_dir=${WAVLM_INPUT_DIR:-${legacy_corpora}/${split}}
output_dir=${WAVLM_OUTPUT_DIR:-${legacy_corpora}/wavlm/${split}}
checkpoint=${WAVLM_CHECKPOINT:-${legacy_corpora}/pretrained_models/wavlm/WavLM-Large.pt}
layer=${WAVLM_LAYER:-6}
batch_size=${WAVLM_BATCH_SIZE:-36}
time_limit=${WAVLM_TIME:-24:00:00}
log_dir=${WAVLM_LOG_DIR:-/exp/xli/ARTS/speech_eval/exp/speechlmscore/wavlm_l6_km50/logs}
split_name=$(basename "${input_dir}")

for path in "${root}" "${input_dir}" "${checkpoint}"; do
  if [ ! -e "${path}" ]; then
    echo "Required path does not exist: ${path}" >&2
    exit 1
  fi
done
if ! [[ "${layer}" =~ ^[1-9][0-9]*$ && "${batch_size}" =~ ^[1-9][0-9]*$ ]]; then
  echo "WAVLM_LAYER and WAVLM_BATCH_SIZE must be positive integers" >&2
  exit 2
fi
mkdir -p "${log_dir}" "${output_dir}"

sbatch \
  --job-name="wavlm_${split_name}_l${layer}" \
  --output="${log_dir}/wavlm_${split_name}_l${layer}_%j.txt" \
  --time="${time_limit}" \
  --export="ALL,WAVLM_ROOT=${root},WAVLM_INPUT_DIR=${input_dir},WAVLM_OUTPUT_DIR=${output_dir},WAVLM_CHECKPOINT=${checkpoint},WAVLM_LAYER=${layer},WAVLM_BATCH_SIZE=${batch_size}" <<'EOT'
#!/usr/bin/env bash
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

if [[ -z "${TMPDIR:-}" || ! -d "${TMPDIR}" ]]; then
  echo "TMPDIR is unavailable; refusing to run without scratch staging" >&2
  exit 1
fi

stage_root="${TMPDIR}/wavlm_extract_${SLURM_JOB_ID}"
cleanup_stage() {
  if [[ -d "${stage_root}" && "${stage_root}" == "${TMPDIR}"/wavlm_extract_* ]]; then
    find "${stage_root}" -depth -delete
  fi
}
trap cleanup_stage EXIT
trap 'exit 143' INT TERM

stage_tree() {
  local source_path=$1
  local destination_path=$2
  mkdir -p "${destination_path}"
  rsync -aL --info=progress2,stats2 "${source_path}/" "${destination_path}/"
}

stage_file() {
  local source_path=$1
  local destination_path=$2
  mkdir -p "$(dirname "${destination_path}")"
  rsync -aL --info=progress2,stats2 "${source_path}" "${destination_path}"
}

mkdir -p "${stage_root}/code" "${stage_root}/input" "${stage_root}/models" "${stage_root}/output"
stage_tree "${WAVLM_ROOT}" "${stage_root}/code"
stage_tree "${WAVLM_INPUT_DIR}" "${stage_root}/input"
stage_file "${WAVLM_CHECKPOINT}" "${stage_root}/models/WavLM.pt"

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "Staged input: ${stage_root}/input"
echo "Persistent output: ${WAVLM_OUTPUT_DIR}"

cd "${stage_root}/code"
python wavlm.py \
  --input-dir "${stage_root}/input" \
  --output-dir "${stage_root}/output" \
  --checkpoint "${stage_root}/models/WavLM.pt" \
  --layer "${WAVLM_LAYER}" \
  --batch-size "${WAVLM_BATCH_SIZE}"

mkdir -p "${WAVLM_OUTPUT_DIR}"
rsync -a --info=progress2,stats2 "${stage_root}/output/" "${WAVLM_OUTPUT_DIR}/"
EOT
