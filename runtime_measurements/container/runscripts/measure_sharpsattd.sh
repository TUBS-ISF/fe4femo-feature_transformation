#!/bin/bash

mkdir /out/tmp/

start=$(date '+%s.%N')

./sharpSAT -decot 120 -decow 100 -tmpdir /out/tmp/ -cs $(echo "$SLURM_MEM_PER_NODE /2 - 500" | bc -l )  "/in/${1}"

end=$(date '+%s.%N')

echo "\n########"
echo "Started at $start, stopped at $end"
runtime=$(bc -l <<< "$end - $start")
echo -e "${runtime}\n########"
