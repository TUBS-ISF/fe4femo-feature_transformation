import argparse
import os
import subprocess
import sys
from pathlib import Path

from helper.input_parser import parse_input

#
# 1. 10-CV
#     - Methode für get-HPO (Model + Selection --> Optuna + static-fold Cross-Val) auf 9-split
#     - Preproc + Trainieren Model auf 9-split mit HPs --> ggf. mit frozen_trial --> alle relevanten Metriken speichern
# 3. Modell mit selbigen wie in CV trainieren auf ganzen Train --> Speichern + HPs
# 4. weitere Eval auf Model --> Speichern

# als separate Main: Multi-Objective für RQ2b


if __name__ == '__main__':
    args = parse_input()
    outputdir = "fe4femo/ml_analysis/out/selector_test/"
    Path(outputdir).mkdir(parents=True, exist_ok=True)
    i = 0
    features = ["all", "prefilter", "SATzilla", "SATfeatPy", "FMBA", "FM_Chara",
                                 "kbest-mutalinfo", "multisurf", "mRMR", "RFE", "harris-hawks",
                                 "genetic", "HFMOEA", "embedded-tree", "SVD-entropy", "NDFS", "optuna-combined"]
    pathData = "raphael-dunkel-master/data/"
    task = "runtime_backbone"
    model = "randomForest"
    hpo_its = "150"

    for feature in features:
        name = f"{args.task}#{args.features}#{args.model}#{args.modelHPO}#{args.HPOits}#{i}"
        arguments = ["sbatch", "--partition=multiple_il", f"--output={outputdir}/{name}.out", "../slurm_scripts/run.sh",
                     "--foldNo", f"{i}", "--features", feature, "--task", task, "--model", model, "--modelHPO", "--HPOits", hpo_its, pathData, outputdir ]
        print(f"Submit fold {i} with arguments:\n {arguments}")
        subprocess.run(arguments)