#!/bin/bash

memLimit=$(echo "$SLURM_MEM_PER_NODE - 500" | bc -l )
coreLimit=$(($SLURM_CPUS_ON_NODE / 2)) # because of hyperthreading on cluster
echo "Running countAntom with $coreLimit threads and $memLimit MB memory"

./countAntom --noThreads=$coreLimit --memSize=$memLimit --preprocessingFileName=/out/preproc.cnf "/in/${1}"
