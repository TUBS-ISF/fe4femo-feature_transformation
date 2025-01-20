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
    outputdir = os.environ.get("HOME") + "/" + args.pathOutput
    Path(outputdir).mkdir(parents=True, exist_ok=True)
    for i in range(10):
        name = f"{args.task}_{args.features}_{args.model}_{args.modelHPO}_{args.HPOits}_{args.foldNo}"
        arguments = ["sbatch", "--partition=multiple_il", f"--output={outputdir}/{name}_{i}.out", "slurm_scripts/run.sh",
                     "--foldNo", f"{i}"] #TODO different time for different combinations

        arguments.extend(sys.argv[1:])
        print(f"Submit fold {i} with arguments:\n {arguments}")
        subprocess.run(arguments)