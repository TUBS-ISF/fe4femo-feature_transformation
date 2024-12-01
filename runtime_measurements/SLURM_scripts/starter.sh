#!/bin/bash

partition="${3:-single}"
maxIt=2334
maxConcurrent=50

mkdir -p $HOME/fe4femo/runtime_measurements/out/$2/

tmpQUEUE=$(squeue | grep -c '^')
waitIt=$((maxConcurrent + 1 - tmpQUEUE))


while [[ maxIt -ge 1 ]] ; do
    while [[ $waitIt -le "-20" ]]
    do
        sleep 30
        queue=$(squeue | grep -c '^')
        free=$((maxConcurrent + 1 - queue))
        echo "$free free spaces in queue"
        waitIt=$free
    done
    ((waitIt=waitIt-1))
    echo "model $maxIt submitted"
    export MODELNO=$maxIt
    output=$(sbatch --output=$HOME/fe4femo/runtime_measurements/out/$2/${maxIt}.out --job-name=${2}_${maxIt} --export=ALL,MODELNO=$maxIt --partition=${partition} $1)
    if [[ $output =~ "error" ]]; then
       echo "Error in Submission, trying again!"
    else
       ((maxIt=maxIt-1))
    fi
done

mail -s "Finished Job Submission \"${2}\"" "raphael.dunkel@uni-ulm.de" <<< "Submission script finished, but jobs could still be running. Check with command 'squeue' to list all currently running jobs!"
