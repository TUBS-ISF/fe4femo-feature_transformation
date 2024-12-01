#!/bin/bash

./spur -seed 42 -t 3600 -cs $(echo "$SLURM_MEM_PER_NODE - 500" | bc -l ) -s ${2} -out /out/samples.out -cnf "/in/${1}"

cat /out/samples.out
