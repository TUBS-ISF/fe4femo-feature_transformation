#!/bin/bash
#SBATCH --time=10:0:0
#SBATCH --job-name=eval_metrics
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32000

export OMP_NUM_THREADS=$((${SLURM_JOB_CPUS_PER_NODE}/2))

perMetricTimeout=3600
container="extractor"

config=$HOME/fe4femo/runtime_measurements/SLURM_scripts/config.txt
inputpath=$HOME/fe4femo/runtime_measurements/feature-model-benchmark/feature_models

no=$MODELNO
pre_input=$(awk -v ArrayTaskID=$no '$1==ArrayTaskID {print $2}' $config)
input=${pre_input}.uvl

echo -e "JOB_ID=${SLURM_JOB_ID}"
echo -e "MODEL_NUMBER=${no}"
echo -e "MODEL_PATH=${input}"

container_path=$HOME/fe4femo/FeatureExtraction/slurm_scripts/${container}_i.sqsh

echo -e "CONTAINER=${container_path}"

echo -e "RERUN=${SLURM_RESTART_COUNT}"

mkdir -p $TMPDIR/in/
mkdir -p $TMPDIR/out/
mkdir -p $TMPDIR/tmp/

if [ -f $inputpath/uvl/$input ]; then
    cp $inputpath/uvl/$input $TMPDIR/in/"${no}".uvl
else
    echo -e "########\nCONTAINER START"
    echo "Missing file"
    echo -e "########\nPROG_STATUS=MISSING_FILE########\n"
    exit 0
fi

echo -e "########\nCONTAINER START"


srun --container-image="$container_path" --container-name=${container}:no_exec \
   --container-mounts=/etc/slurm/task_prolog:/etc/slurm/task_prolog,/scratch:/scratch,$TMPDIR/in:/in,$TMPDIR/out:/out,$TMPDIR/tmp:/tmp \
   --container-workdir=/app/ --no-container-entrypoint java -jar -Djava.io.tmpdir=$TMPDIR fe.jar /in/"${no}".uvl $perMetricTimeout
retValue=$?

if [[ ${retValue} -eq 0 ]]; then
    echo -e "###########\PROG_STATUS=SUCCESS\n###########\n"
    exit 0
else
    if [[ ${SLURM_RESTART_COUNT} -lt 0 ]]; then # Deactivated
        scontrol requeue $SLURM_JOB_ID
        exit 1
    else
        echo -e "###########\PROG_STATUS=ERROR\n###########\n"
        exit -1
    fi
fi

