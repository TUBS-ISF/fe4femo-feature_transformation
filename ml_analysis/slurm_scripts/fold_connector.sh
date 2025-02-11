#!/bin/bash

cd /app || exit 1

for ((i = 0 ; i < 10; i++ )); do
  echo "$(date +"%Y-%m-%d %H:%M:%S.%N")   Starting fold $i"
  conda run --no-capture-output -n ml_analysis python -Wignore::FutureWarning generate_fold_model.py "--foldNo" "$i" "$@"
  echo "$(date +"%Y-%m-%d %H:%M:%S.%N")   Finished fold $i"
done