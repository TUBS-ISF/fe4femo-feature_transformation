#!/bin/bash

file=/in/${1}
solving_file=$(mktemp --tmpdir=/out XXXXX.cnf)
clean_file=$(mktemp --tmpdir=/out XXXXX.cnf)
preprocessed_file=$(mktemp --tmpdir=/out XXXXX.cnf)
cache_size=$SLURM_MEM_PER_NODE
tout_be=1800
timeout_total=3600

mc=$(grep "^c t " $file)
echo "c o found header: $mc"

echo "c o file: ${file} clean_file: ${clean_file} preprocessed_file: ${preprocessed_file} solving_file: ${solving_file}"

start=$(date '+%s.%N')

time_begin=$(date +%s)
timeout ${tout_be} ./arjun ${file} ${preprocessed_file} > /dev/null 2>&1
preprocessing_status=$(grep "^p cnf" ${preprocessed_file})
if [[ ${preprocessing_status} == *"p cnf"* ]]; then
   echo "c o OK, Arjun succeeded"
   grep -v "^c" ${preprocessed_file} > ${clean_file}
   multi=`grep "^c MUST MULTIPLY BY" ${preprocessed_file} | awk '{print $5}'`
else
   echo "c o WARNING Arjun did NOT succeed"
   multi=1
fi
echo "c o MULTI will be ${multi}"

time_end=$(date +%s)

timeout_mc=$((timeout_total + time_begin - time_end))
total_mem_gb=$(echo "$SLURM_MEM_PER_NODE / 1024" | bc -l)
echo "c o Running ExactMC, timeleft: ${timeout_mc} seconds, memo: ${total_mem_gb} GB"

./ExactMC --competition --memo $total_mem_gb ${clean_file} 2>&1 | tee ${solving_file}

state=$(grep "^s" ${solving_file})
count=$(grep "^c o Number of models:" ${solving_file} | awk '{print $6}')

export BC_LINE_LENGTH=1000000000
tuned_count=$(echo "${count}*${multi}" | bc -l)
if [[ ${tuned_count} == "0" ]]; then log_10_count="-inf"
else log_10_count=$(echo "scale=15; l($tuned_count)/l(10)" | bc -l)
fi

echo $state
echo "c s type mc"
echo "c s log10-estimate ${log_10_count}"
echo "c s exact arb int ${tuned_count}"
