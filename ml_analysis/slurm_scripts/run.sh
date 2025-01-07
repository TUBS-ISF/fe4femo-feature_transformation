#!/bin/bash
#SBATCH --time=10:0:0
#SBATCH --job-name=eval_model
#SBATCH --ntasks=40
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=1125
#SBATCH --nodes=2,7ß


export OMP_NUM_THREADS=$((${SLURM_CPUS_PER_TASK}/2))

container="ml_analysis"

inputpath=$HOME/fe4femo/runtime_measurements/feature-model-benchmark/feature_models

echo -e "JOB_ID=${SLURM_JOB_ID}"

container_path=$HOME/fe4femo/FeatureExtraction/slurm_scripts/${container}_i.sqsh
mkdir -p $TMPDIR/in/
mkdir -p $TMPDIR/out/
mkdir -p $TMPDIR/tmp/


echo -e "CONTAINER=${container_path}"
echo -e "########\nCONTAINER START"


srun --container-image="$container_path" --container-name=${container}:no_exec \
   --container-mounts=/etc/slurm/task_prolog:/etc/slurm/task_prolog,/scratch:/scratch,$TMPDIR/in:/in,$TMPDIR/out:/out,$TMPDIR/tmp:/tmp \
   --container-workdir=/app/ --no-container-entrypoint conda run --no-capture-output -n ml_analysis python generate_fold_model.py "$@"

