#!/bin/bash

start=$(date '+%s.%N')

mkdir /out/tmp/

./d4 -m counting -i "/in/${1}"

end=$(date '+%s.%N')

echo "\n########"
echo "Started at $start, stopped at $end"
runtime=$(bc -l <<< "$end - $start")
echo -e "${runtime}\n########"
