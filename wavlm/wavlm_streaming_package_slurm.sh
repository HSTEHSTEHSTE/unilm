#!/usr/bin/env bash

# Package streaming WavLM features in three sequential, remotely handed-off blobs.
# The remote receiver must copy each .tar.gz and then create its matching
# .received marker in the package directory.  The job deletes an archive only
# after that marker is present.
#
#   ./wavlm_streaming_package_slurm.sh
#   ./wavlm_streaming_package_slurm.sh 5-00:00:00
#   ./wavlm_streaming_package_slurm.sh 5-00:00:00 cpu

set -euo pipefail

if [ "$#" -gt 2 ]; then
  echo "Usage: $0 [time-limit] [partition]" >&2
  exit 2
fi

time_limit=${1:-5-00:00:00}
partition=${2:-}
if ! [[ "${time_limit}" =~ ^([0-9]+-)?[0-9]{1,2}:[0-9]{2}:[0-9]{2}$ ]]; then
  echo "Time limit must use [days-]hours:minutes:seconds format: ${time_limit}" >&2
  exit 2
fi

sbatch_args=(
  --time="${time_limit}"
  --job-name=wavlm_streaming_package
  --output=wavlm_streaming_package_%j.txt
  --cpus-per-task=16
  --mem=32G
)
if [ -n "${partition}" ]; then
  sbatch_args+=(--partition="${partition}")
fi

sbatch "${sbatch_args[@]}" <<'EOT'
#!/usr/bin/env bash
#
#SBATCH --nodes=1
#SBATCH --ntasks=1

set -euo pipefail

feature_root=/exp/xli/Voice-Privacy-Challenge-2024/corpora/wavlm_streaming/step5_history-1_lookahead10/LibriSpeech
package_dir=/exp/xli/Voice-Privacy-Challenge-2024/corpora/wavlm_streaming/packages/step5_history-1_lookahead10
poll_seconds=60

mkdir -p "${package_dir}"

available_bytes() {
  df -PB1 "${package_dir}" | awk 'NR == 2 {print $4}'
}

require_free_space() {
  local required_bytes=$1
  local available
  available=$(available_bytes)
  if [ "${available}" -lt "${required_bytes}" ]; then
    echo "Insufficient space in ${package_dir}: need ${required_bytes} bytes, have ${available}." >&2
    exit 1
  fi
}

make_archive() {
  local archive_path=$1
  shift

  local temporary_path="${archive_path}.tmp"
  rm -f "${temporary_path}"

  if command -v pigz >/dev/null 2>&1; then
    tar -C "${feature_root}" \
      --use-compress-program="pigz -p ${SLURM_CPUS_PER_TASK:-1}" \
      -cf "${temporary_path}" -- "$@"
  else
    tar -C "${feature_root}" -czf "${temporary_path}" -- "$@"
  fi
  mv "${temporary_path}" "${archive_path}"
}

package_blob() {
  local label=$1
  local minimum_free_gib=$2
  shift 2

  local archive_path="${package_dir}/${label}.tar.gz"
  local ready_path="${archive_path}.ready"
  local receipt_path="${archive_path}.received"
  local checksum_path="${archive_path}.sha256"
  local required_bytes=$((minimum_free_gib * 1024 * 1024 * 1024))
  local source_path

  for source_path in "$@"; do
    if [ ! -d "${feature_root}/${source_path}" ]; then
      echo "Missing feature directory: ${feature_root}/${source_path}" >&2
      exit 1
    fi
  done

  # A previous run may have been restarted while waiting for remote receipt.
  # In that case, do not recreate or overwrite the archive.
  if [ -f "${receipt_path}" ]; then
    rm -f "${archive_path}" "${ready_path}" "${checksum_path}" "${archive_path}.tmp"
    echo "${label}: receipt already exists; cleaned any retained local archive."
    return
  fi

  if [ -f "${ready_path}" ]; then
    if [ ! -f "${archive_path}" ]; then
      echo "${label}: ready marker exists but archive is missing: ${archive_path}" >&2
      exit 1
    fi
    echo "${label}: archive already ready; waiting for remote receipt."
  else
    if [ -f "${archive_path}" ]; then
      echo "${label}: removing archive without a ready marker before rebuilding."
      rm -f "${archive_path}" "${checksum_path}"
    fi

    require_free_space "${required_bytes}"
    echo "${label}: creating ${archive_path} from: $*"
    make_archive "${archive_path}" "$@"
    sha256sum "${archive_path}" > "${checksum_path}.tmp"
    mv "${checksum_path}.tmp" "${checksum_path}"
    {
      echo "archive=$(basename "${archive_path}")"
      echo "sha256_file=$(basename "${checksum_path}")"
      echo "created_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    } > "${ready_path}.tmp"
    mv "${ready_path}.tmp" "${ready_path}"
    echo "${label}: archive ready. Awaiting ${receipt_path}."
  fi

  while [ ! -f "${receipt_path}" ]; do
    sleep "${poll_seconds}"
  done

  rm -f "${archive_path}" "${ready_path}" "${checksum_path}"
  echo "${label}: receipt found; removed local archive and proceeding."
}

# These run in order. At most one archive is retained locally at a time.
package_blob train-other-500 400 train-other-500
package_blob train-clean-360 300 train-clean-360
package_blob remaining-splits 120 \
  dev-clean dev-other test-clean test-other train-clean-100

echo "All streaming WavLM blobs were acknowledged and cleared locally."
EOT
