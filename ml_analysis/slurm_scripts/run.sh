#!/bin/bash
#SBATCH --time=6:30:0
#SBATCH --job-name=eval_model
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=1950
#SBATCH --nodes=2-70
#SBATCH --use-min-nodes


export OMP_NUM_THREADS=$(( ${SLURM_CPUS_PER_TASK} - 1 < 1 ? 1 : ${SLURM_CPUS_PER_TASK} - 1 ))
echo -e "JOB_ID=${SLURM_JOB_ID}"
echo -e "OMP_THREADS=${OMP_NUM_THREADS}"

container="ml_analysis"

mkdir -p $TMPDIR/in/
mkdir -p $TMPDIR/out/
mkdir -p $TMPDIR/tmp/


echo -e "########\nCONTAINER START"
ENROOT_CONFIG_PATH=$HOME/enroot_config/

# helper: srun --container-image=$HOME/fe4femo/ml_analysis/slurm_scripts/ml_analysis_i.sqsh --container-name=ml_analysis:no_exec    --container-mounts=/etc/slurm/task_prolog:/etc/slurm/task_prolog,/scratch:/scratch    --container-workdir=/app/ --time=10 --partition=dev_single --no-container-entrypoint /bin/bash
srun --exact --container-image=ghcr.io#rsd6170/ml_analysis:0.1 --container-name=ml_analysis:no_exec \
   --container-mounts=/etc/slurm/task_prolog:/etc/slurm/task_prolog,/scratch:/scratch  --container-mount-home \
   --container-workdir=/app/ --no-container-entrypoint conda run --no-capture-output -n ml_analysis python -Wignore::FutureWarning generate_fold_model.py $@

