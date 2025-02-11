#!/bin/bash
#SBATCH --time=1:0:0
#SBATCH --job-name=eval_model
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=128
#SBATCH --mem-per-cpu=1950
#SBATCH --nodes=2-70
#SBATCH --use-min-nodes

if [[ -z "$ML_FOLD" ]]; then
  echo "Must provide ML_FOLD in environment" 1>&2
  exit 1
fi

export DASK_LOGGING__DISTRIBUTED=WARN

export OMP_NUM_THREADS=2
echo -e "JOB_ID=${SLURM_JOB_ID}"
echo -e "OMP_THREADS=${OMP_NUM_THREADS}"

echo -e "########\nCONTAINER START"
ENROOT_CONFIG_PATH=$HOME/enroot_config/

# helper: srun --container-image=$HOME/fe4femo/ml_analysis/slurm_scripts/ml_analysis_i.sqsh --container-name=ml_analysis:no_exec    --container-mounts=/etc/slurm/task_prolog:/etc/slurm/task_prolog,/scratch:/scratch    --container-workdir=/app/ --time=10 --partition=dev_single --no-container-entrypoint /bin/bash

if [ "$ML_FOLD" = "-1" ]; then
  RUN_COMMAND=("/app/slurm_scripts/fold_connector.sh")
else
  RUN_COMMAND=("conda" "run" "--no-capture-output" "-n" "ml_analysis" "python" "-Wignore::FutureWarning" "generate_fold_model.py" "--foldNo" "$ML_FOLD")
fi



srun --exact --container-image=ghcr.io#rsd6170/ml_analysis:0.1 --container-name=ml_analysis:no_exec \
   --container-mounts=/etc/slurm/task_prolog:/etc/slurm/task_prolog,/scratch:/scratch  --container-mount-home \
   --container-workdir=/app/ --no-container-entrypoint "${RUN_COMMAND[@]}" "$@"

