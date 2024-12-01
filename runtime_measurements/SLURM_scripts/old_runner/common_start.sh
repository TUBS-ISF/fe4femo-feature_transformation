config=$HOME/feature_metrics/performance/config.txt
inputpath=$HOME/feature_metrics/performance/feature-model-benchmark/feature_models

no=$(bc <<< "$SLURM_ARRAY_TASK_ID + (96 * $OWNmodifier)")
input=$(awk -v ArrayTaskID=$no '$1==ArrayTaskID {print $2}' $config)

echo -e "MODEL_NUMBER=${no}"
echo -e "MODEL_PATH=${input}"

container_path=$HOME/feature_metrics/performance/enroot/$(echo "${1}_i.sqsh")

echo -e "CONTAINER=${container_path}"

echo -e "RERUN=${SLURM_RESTART_COUNT}"

mkdir -p $TMPDIR/in/
mkdir -p $TMPDIR/out/

if [ -f $inputpath/dimacs/$input ]; then
    cp $inputpath/dimacs/$input $TMPDIR/in/input.dimacs
elif [ -f $inputpath/original/$input ]; then
    cp $inputpath/original/$input $TMPDIR/in/input.dimacs
elif [ -f $inputpath/uvl/$input ]; then
    cp $inputpath/uvl/$input $TMPDIR/in/input.dimacs
else
    echo -e "########\nCONTAINER START"
    echo "Missing file"
    echo -e "########\n3601.1337########\n"
    exit 0
fi

echo -e "########\nCONTAINER START"

