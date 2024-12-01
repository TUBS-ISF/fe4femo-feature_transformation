#!/bin/bash

start=$(date '+%s.%N')

./kissat "/in/${1}"

end=$(date '+%s.%N')

echo "\n########"
echo "Started at $start, stopped at $end"
runtime=$(bc -l <<< "$end - $start")
echo -e "${runtime}\n########"
