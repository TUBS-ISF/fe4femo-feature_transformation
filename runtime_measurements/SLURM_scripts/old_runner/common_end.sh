retValue=$1

if [[ ${retValue} -eq 0 ]]; then
    exit 0
elif  [[ ${retValue} -eq 124 ]]; then
    echo -e "###########\n3600.42\n###########\n"
    exit 0
else
    if [[ ${SLURM_RESTART_COUNT} -le 2 ]]; then
        scontrol requeue $SLURM_JOB_ID
        exit 1
    else
        echo -e "###########\n3600.999999\n###########\n"
        exit -1
    fi
fi
