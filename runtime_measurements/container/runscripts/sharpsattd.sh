#!/bin/bash

mkdir /out/tmp/

./sharpSAT -decot 120 -decow 100 -tmpdir /out/tmp/ -cs $(echo "$SLURM_MEM_PER_NODE / 2 - 500" | bc -l )  "/in/${1}"
