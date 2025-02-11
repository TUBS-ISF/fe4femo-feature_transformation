#!/bin/bash

cd /app || exit 1

echo "$(date +"%Y-%m-%d %H:%M:%S.%N")   Starting fold $ML_FOLD"
conda run --no-capture-output -n ml_analysis python -Wignore::FutureWarning generate_fold_model.py "--foldNo" "$ML_FOLD" "$@"
echo "$(date +"%Y-%m-%d %H:%M:%S.%N")   Finished fold $ML_FOLD"