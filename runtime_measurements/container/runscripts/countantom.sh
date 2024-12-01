#!/bin/bash

memLimit=$(echo "$SLURM_MEM_PER_NODE - 500" | bc -l )
echo "Running countAntom with $SLURM_CPUS_ON_NODE threads and $memLimit MB memory"

./countAntom --noThreads=$SLURM_CPUS_ON_NODE --memSize=$memLimit --preprocessingFileName=/out/preproc.cnf "/in/${1}"
