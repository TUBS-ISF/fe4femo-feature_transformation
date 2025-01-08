#!/bin/bash
#SBATCH --time=3:0:0
#SBATCH --job-name=eval_model
#SBATCH --ntasks=80
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=1125
#SBATCH --nodes=2-70
#SBATCH --use-min-nodes


export OMP_NUM_THREADS=$((${SLURM_CPUS_PER_TASK}/2))
echo -e "JOB_ID=${SLURM_JOB_ID}"

container="ml_analysis"
container_path=$HOME/fe4femo/ml_analysis/slurm_scripts/${container}_i.sqsh

mkdir -p $TMPDIR/in/
mkdir -p $TMPDIR/out/
mkdir -p $TMPDIR/tmp/


echo -e "CONTAINER=${container_path}"
echo -e "########\nCONTAINER START"

# helper: srun --container-image=$HOME/fe4femo/ml_analysis/slurm_scripts/ml_analysis_i.sqsh --container-name=ml_analysis:no_exec    --container-mounts=/etc/slurm/task_prolog:/etc/slurm/task_prolog,/scratch:/scratch    --container-workdir=/app/ --time=10 --partition=dev_single --no-container-entrypoint /bin/bash
srun --exact --container-image="$container_path" --container-name=${container}:no_exec \
   --container-mounts=/etc/slurm/task_prolog:/etc/slurm/task_prolog,/scratch:/scratch  --container-mount-home \
   --container-workdir=/app/ --no-container-entrypoint conda run --no-capture-output -n ml_analysis python generate_fold_model.py $@

