#!/usr/bin/env bash

# Submit a CPU-only job that removes offline (non-streaming) WavLM training
# features. It does not touch corpora/wavlm_streaming.
#
#   ./wavlm_remove_training_slurm.sh --confirm

set -euo pipefail

if [ "$#" -ne 1 ] || [ "$1" != '--confirm' ]; then
  echo "This permanently removes offline WavLM training features." >&2
  echo "Usage: $0 --confirm" >&2
  exit 2
fi

sbatch \
  --job-name=remove_wavlm_training \
  --output=remove_wavlm_training_%j.txt \
  --time=24:00:00 \
  --cpus-per-task=1 \
  --mem=2G <<'EOT'
#!/usr/bin/env bash
#
#SBATCH --nodes=1
#SBATCH --ntasks=1

set -euo pipefail

feature_root=/exp/xli/Voice-Privacy-Challenge-2024/corpora/wavlm/LibriSpeech
splits=(train-clean-100 train-clean-360 train-other-500)

if [ ! -d "${feature_root}" ]; then
  echo "Offline WavLM root does not exist: ${feature_root}" >&2
  exit 1
fi

resolved_root=$(realpath -e "${feature_root}")

for split in "${splits[@]}"; do
  target="${feature_root}/${split}"
  if [ ! -d "${target}" ]; then
    echo "Already absent, skipping: ${target}"
    continue
  fi

  if [ "$(realpath -e "${target}/..")" != "${resolved_root}" ]; then
    echo "Refusing unexpected deletion target: ${target}" >&2
    exit 1
  fi

  echo "Removing: ${target}"
  rm -rf --one-file-system -- "${target}"

  if [ -e "${target}" ]; then
    echo "Deletion did not complete: ${target}" >&2
    exit 1
  fi
done

echo "Offline WavLM training features removed."
EOT
