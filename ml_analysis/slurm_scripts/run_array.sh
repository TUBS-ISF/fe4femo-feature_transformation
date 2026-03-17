#!/bin/bash
#SBATCH --job-name=eval_model_array
#SBATCH --cpus-per-task=128
#SBATCH --mem-per-cpu=1950
#SBATCH --nodes=1
set -euo pipefail

if [[ "${#}" -lt 3 ]]; then
  echo "Usage: $0 <config_path> <data_path> <output_path>" 1>&2
  exit 1
fi

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "SLURM_ARRAY_TASK_ID is required" 1>&2
  exit 1
fi

config_path="$1"
data_path="$2"
output_path="$3"
config_offset="${CONFIG_OFFSET:-0}"
config_id=$((config_offset + SLURM_ARRAY_TASK_ID))
line_no=$((config_id + 2))

if [[ "${data_path}" != /* ]]; then
  data_path="$HOME/${data_path}"
fi
if [[ "${output_path}" != /* ]]; then
  output_path="$HOME/${output_path}"
fi

if [[ ! -f "${config_path}" ]]; then
  echo "Config file not found: ${config_path}" 1>&2
  exit 1
fi

line="$(sed -n "${line_no}p" "${config_path}")"
if [[ -z "${line}" ]]; then
  echo "No config row for array index ${SLURM_ARRAY_TASK_ID} with offset ${config_offset} (config id ${config_id}, line ${line_no})" 1>&2
  exit 1
fi

read -r experiment_no name task_count runtime fold_no feature task model hpo_its bool_model_hpo bool_selector_hpo bool_multi_objective transformation <<< "${line}"

if [[ "${experiment_no}" != "${config_id}" ]]; then
  echo "Warning: config id ${experiment_no} does not match resolved config id ${config_id} (array ${SLURM_ARRAY_TASK_ID}, offset ${config_offset})" 1>&2
fi

if [[ "${bool_model_hpo}" == "True" ]]; then
  model_hpo_flag="--modelHPO"
else
  model_hpo_flag="--no-modelHPO"
fi

if [[ "${bool_selector_hpo}" == "True" ]]; then
  selector_hpo_flag="--selectorHPO"
else
  selector_hpo_flag="--no-selectorHPO"
fi

if [[ "${bool_multi_objective}" == "True" ]]; then
  multi_objective_flag="--multiObjective"
else
  multi_objective_flag="--no-multiObjective"
fi

hard_root="${HARD_ROOT:-$HOME/fe4femo-feature_transformation}"
sif_path="${SIF_PATH:-${hard_root}/ml_analysis/ml_analysis_ft_v1.sif}"

export DASK_LOGGING__DISTRIBUTED=WARN
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore}"
export JOBLIB_TEMP_FOLDER="${JOBLIB_TEMP_FOLDER:-${TMPDIR:-/tmp}}"
export TMPDIR="${TMPDIR:-/tmp}"

echo "JOB_ID=${SLURM_JOB_ID}"
echo "ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
echo "CONFIG_OFFSET=${config_offset}"
echo "EXPERIMENT_ID=${experiment_no}"
echo "RUN_NAME=${name}"
echo "RUNTIME_MIN=${runtime}"
echo "FEATURE=${feature} MODEL=${model} TASK=${task} FOLD=${fold_no} TRANSFORMATION=${transformation}"
echo "SIF_PATH=${sif_path}"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1 -> skipping srun execution"
  exit 0
fi

if [[ ! -f "${sif_path}" ]]; then
  echo "SIF image not found: ${sif_path}" 1>&2
  exit 1
fi

# Copy SIF to node-local scratch to avoid squashfuse failures over NFS.
local_sif="${TMPDIR:-/tmp}/ml_analysis_ft_v1.sif"
echo "Copying SIF to local scratch: ${local_sif}"
cp "${sif_path}" "${local_sif}"
sif_path="${local_sif}"

data_arg="${data_path}"
output_arg="${output_path}"

# The container entrypoint prefixes HOME internally, so pass HOME-relative paths.
if [[ "${data_arg}" == "$HOME/"* ]]; then
  data_arg="${data_arg#"$HOME/"}"
fi
if [[ "${output_arg}" == "$HOME/"* ]]; then
  output_arg="${output_arg#"$HOME/"}"
fi

srun \
  --exact \
  -n 1 \
  /usr/bin/apptainer run \
  --bind "$HOME:$HOME" \
  "${sif_path}" \
  --foldNo "${fold_no}" \
  --features "${feature}" \
  --task "${task}" \
  --model "${model}" \
  --HPOits "${hpo_its}" \
  "${model_hpo_flag}" \
  "${selector_hpo_flag}" \
  "${multi_objective_flag}" \
  --transformation "${transformation}" \
  "${data_arg}" \
  "${output_arg}"
