#!/bin/bash

partition="${2:-single}"
maxIt=24

for i in $(seq 0 ${maxIt});
do
    until [ "$(squeue | grep -c '^')" -eq "1"  ]
    do
        echo "wait until finished"
        sleep 30
    done
    echo "loop $i starts"
    export OWNmodifier=$i
    MYmail=$([ $i -eq ${maxIt} ] && echo "--mail-type=END --mail-user=\"raphael.dunkel@uni-ulm.de\"" || echo " " )
    sbatch --output=${i}_%a.out --export=ALL,OWNmodifier=$i ${MYmail} --partition=${partition} $1
    sleep 30
done
