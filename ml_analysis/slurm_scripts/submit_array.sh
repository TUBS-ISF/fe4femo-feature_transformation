#!/bin/bash
set -euo pipefail

partition="${1:-multiple_il}"
max_concurrent="${4:-50}"
# Pragmatic hardcoded root for HPC runs; override with HARD_ROOT if needed.
hard_root="${HARD_ROOT:-$HOME/fe4femo-feature_transformation}"
sif_path="${6:-${hard_root}/ml_analysis/ml_analysis_ft_v1.sif}"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

output_path="${2:-${hard_root}/ml_analysis/out/main}"
config="${3:-${hard_root}/ml_analysis/config.txt}"
data_path="${5:-${hard_root}/data}"

run_script="${script_dir}/run_array.sh"

if [[ ! -f "${config}" ]]; then
  echo "Config file not found: ${config}" 1>&2
  exit 1
fi

total_experiments=$(awk 'NR>1 && $1 ~ /^[0-9]+$/ {c++} END {print c+0}' "${config}")
if [[ "${total_experiments}" -le 0 ]]; then
  echo "Config file has no experiments: ${config}" 1>&2
  exit 1
fi

max_array_id=$(awk 'NR>1 && $1 ~ /^[0-9]+$/ {m=$1} END {print m+0}' "${config}")
max_runtime=$(awk 'NR>1 && $1 ~ /^[0-9]+$/ { if ($4 > m) m = $4 } END { print m+0 }' "${config}")
max_tasks=$(awk 'NR>1 && $1 ~ /^[0-9]+$/ { if ($3 > m) m = $3 } END { print m+0 }' "${config}")

if [[ "${max_runtime}" -le 0 ]]; then
  max_runtime=120
fi
if [[ "${max_tasks}" -le 0 ]]; then
  max_tasks=3
fi

# Add a small buffer to reduce job timeout due to minor variance.
time_limit=$((max_runtime + 10))

mkdir -p "${output_path}"

echo "Submitting array 0-${max_array_id}%${max_concurrent} to partition ${partition}"
echo "Using time limit ${time_limit} minutes and ${max_tasks} tasks per job"
echo "Config: ${config}"
echo "Data path: ${data_path}"
echo "Output path: ${output_path}"
echo "SIF path: ${sif_path}"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1 -> skipping sbatch submission"
  exit 0
fi

sbatch \
  --partition="${partition}" \
  --array="0-${max_array_id}%${max_concurrent}" \
  --time="${time_limit}" \
  --ntasks="${max_tasks}" \
  --output="${output_path}/slurm_%A_%a.out" \
  --export=ALL,SIF_PATH="${sif_path}" \
  "${run_script}" "${config}" "${data_path}" "${output_path}"
