#!/bin/bash
#SBATCH --time=125
#SBATCH --job-name=eval_metrics
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=9500

perMetricTimeout=120
container="extractor"

config=$HOME/fe4femo/runtime_measurements/SLURM_scripts/config.txt
inputpath=$HOME/fe4femo/runtime_measurements/feature-model-benchmark/feature_models

no=$MODELNO
pre_input=$(awk -v ArrayTaskID=$no '$1==ArrayTaskID {print $2}' $config)
input=${pre_input}.uvl

echo -e "JOB_ID=${SLURM_JOB_ID}"
echo -e "MODEL_NUMBER=${no}"
echo -e "MODEL_PATH=${input}"

container_path=$HOME/fe4femo/runtime_measurements/container/enroot/${container}_i.sqsh

echo -e "CONTAINER=${container_path}"

echo -e "RERUN=${SLURM_RESTART_COUNT}"

mkdir -p $TMPDIR/in/
mkdir -p $TMPDIR/out/

if [ -f $inputpath/uvl/$input ]; then
    cp $inputpath/uvl/$input $TMPDIR/in/$input
else
    echo -e "########\nCONTAINER START"
    echo "Missing file"
    echo -e "########\n3601.1337########\n"
    exit 0
fi

echo -e "########\nCONTAINER START"



timeout 3600 srun --container-image=${container_path}  --container-name=${container}:no_exec \
   --container-mounts=/etc/slurm/task_prolog:/etc/slurm/task_prolog,/scratch:/scratch,$TMPDIR/in:/in,$TMPDIR/out:/out \
   --container-workdir=/app/ --no-container-entrypoint java -jar -Djava.io.tmpdir=/out/ fe.jar /in/$input $perMetricTimeout
retValue=$?

if [[ ${retValue} -eq 0 ]]; then
    exit 0
elif  [[ ${retValue} -eq 124 ]]; then
    echo -e "###########\n3600.42\n###########\n"
    exit 0
else
    if [[ ${SLURM_RESTART_COUNT} -le 1 ]]; then
        scontrol requeue $SLURM_JOB_ID
        exit 1
    else
        echo -e "###########\n3600.999999\n###########\n"
        exit -1
    fi
fi

