#!/bin/bash

start=$(date '+%s.%N')

./spur -seed 42 -t 3600 -cs $(echo "$SLURM_MEM_PER_NODE - 500" | bc -l ) -s ${2} -out /out/samples.out -cnf "/in/${1}"

end=$(date '+%s.%N')

cat /out/samples.out

echo "\n########"
echo "Started at $start, stopped at $end"
runtime=$(bc -l <<< "$end - $start")
echo -e "${runtime}\n########"
