#!/bin/bash
#SBATCH --time=65
#SBATCH --job-name=eval_countAntom
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=15750

container="countantom"

source $HOME/feature_metrics/performance/common_start.sh $container

timeout 3600 srun --container-image=${container_path}  --container-name=${container}:no_exec \
   --container-mounts=/etc/slurm/task_prolog:/etc/slurm/task_prolog,/scratch:/scratch,$TMPDIR/in:/in,$TMPDIR/out:/out \
  --no-container-entrypoint /app/measure.sh input.dimacs
retValue=$?

source $HOME/feature_metrics/performance/common_end.sh $retValue
