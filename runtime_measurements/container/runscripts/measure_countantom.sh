#!/bin/bash

memLimit=$(echo "$SLURM_MEM_PER_NODE - 500" | bc -l )
echo "Running countAntom with $SLURM_CPUS_ON_NODE threads and $memLimit MB memory"

start=$(date '+%s.%N')

./countAntom --noThreads=$SLURM_CPUS_ON_NODE --memSize=$memLimit --preprocessingFileName=/out/preproc.cnf "/in/${1}"

end=$(date '+%s.%N')

echo "\n########"
echo "Started at $start, stopped at $end"
runtime=$(bc -l <<< "$end - $start")
echo -e "${runtime}\n########"
