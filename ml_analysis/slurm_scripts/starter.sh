#!/bin/bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
repo_name="$(basename "${repo_root}")"

partition="${1:-multiple_il}"
output_path="${2:-${repo_name}/ml_analysis/out/main}"
config="${3:-${repo_root}/ml_analysis/config.txt}"
data_path="${4:-${repo_name}/data/}"
script_path="${script_dir}/run.sh"

mkdir -p "$HOME/$output_path"

maxIt=$(awk 'NR>1 && $1 ~ /^[0-9]+$/ {m=$1} END {print m+0}' "$config")
maxConcurrent=50

tmpQUEUE=$(squeue | grep -c '^')
waitIt=$((maxConcurrent + 1 - tmpQUEUE))

while [[ $maxIt -ge 0 ]]; do
    while [[ $waitIt -le "-20" ]]; do
        sleep 120
        queue=$(squeue | grep -c '^')
        free=$((maxConcurrent + 1 - queue))
        echo "$free free spaces in queue"
        waitIt=$free
    done
    ((waitIt=waitIt-1))
    echo "model $maxIt submitted"
    EXPERIMENT_NO=$maxIt

    name=$(awk -v ID=$EXPERIMENT_NO '$1==ID {print $2}' "$config")
    task_count=$(awk -v ID=$EXPERIMENT_NO '$1==ID {print $3}' "$config")
    runtime=$(awk -v ID=$EXPERIMENT_NO '$1==ID {print $4}' "$config")
    out_path="$HOME/${output_path}/${name}.out"
    fold_no=$(awk -v ID=$EXPERIMENT_NO '$1==ID {print $5}' "$config")
    feature=$(awk -v ID=$EXPERIMENT_NO '$1==ID {print $6}' "$config")
    task=$(awk -v ID=$EXPERIMENT_NO '$1==ID {print $7}' "$config")
    model=$(awk -v ID=$EXPERIMENT_NO '$1==ID {print $8}' "$config")
    HPOits=$(awk -v ID=$EXPERIMENT_NO '$1==ID {print $9}' "$config")
    bool_modelHPO=$(awk -v ID=$EXPERIMENT_NO '$1==ID {print $10}' "$config")
    [[ "$bool_modelHPO" == "True" ]] && modelHPO="--modelHPO" || modelHPO="--no-modelHPO"
    bool_selectorHPO=$(awk -v ID=$EXPERIMENT_NO '$1==ID {print $11}' "$config")
    [[ "$bool_selectorHPO" == "True" ]] && selectorHPO="--selectorHPO" || selectorHPO="--no-selectorHPO"
    bool_multiObjective=$(awk -v ID=$EXPERIMENT_NO '$1==ID {print $12}' "$config")
    [[ "$bool_multiObjective" == "True" ]] && multiObjective="--multiObjective" || multiObjective="--no-multiObjective"
    transformation=$(awk -v ID=$EXPERIMENT_NO '$1==ID {print $13}' "$config")

    output=$(sbatch -J "$name" -n "$task_count" --time="$runtime" --output="$out_path" --partition="${partition}" --export=ALL,ML_FOLD="$fold_no" "$script_path" --features "$feature" --task "$task" --model "$model" --HPOits "$HPOits" "$modelHPO" "$selectorHPO" "$multiObjective" --transformation "$transformation" "$data_path" "$output_path")
    if [[ $output =~ "error" ]]; then
       echo "Error in Submission, trying again!"
    else
       ((maxIt=maxIt-1))
    fi
done

mail -s "Finished Job Submission" "raphael.dunkel@uni-ulm.de" <<< "Submission script finished, but jobs could still be running. Check with command 'squeue' to list all currently running jobs!"
