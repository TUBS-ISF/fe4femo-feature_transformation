#!/bin/bash

declare -a arr=("approxmc" "cadiback" "countantom" "d4v2_23" "d4v2_24" "exactmc_arjun" "ganak" "kissat" "sharpsattd" "spur")

rm -rf enroot/

mkdir enroot

for i in "${arr[@]}"
do
   echo "Building $i"
   docker build -t $i --file "${i}.docker" .
   enroot import -o "enroot/${i}_i.sqsh" "dockerd://${i}"
done
