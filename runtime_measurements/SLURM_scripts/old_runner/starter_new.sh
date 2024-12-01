#!/bin/bash

partition="${1:-single}" #TODO insert default partition
maxIt=1879
maxConcurrent=50 #TODO change max concurrent


waitIt=$maxConcurrent

while [[ maxIt -ge 1 ]] ; do
    while [[ $waitIt -le "-10" ]]
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
    output=$(sbatch --output=./out/${maxIt}_%j.out --export=ALL,MODELNO=$maxIt --partition=${partition} history_eval.sh)
    if [[ $output =~ "error" ]]; then
       echo "Error in Submission, trying again!"
    else
       ((maxIt=maxIt-1))
    fi
done

echo "Submission script finished, but jobs could still be running. Check with command 'squeue' to list all currently running jobs!" | mail -s "Finished Job Submission on SLURM Cluster" "mailto@changehere" #TODO insert your mail
