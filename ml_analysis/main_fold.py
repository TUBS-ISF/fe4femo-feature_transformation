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

def main(pathOutput : str):
    outputdir = os.environ.get("HOME")+"/"+pathOutput
    Path(outputdir).mkdir(parents=True, exist_ok=True)
    for i in range(10):
        arguments = ["sbatch", "--partition=multiple_il", f"--output={outputdir}/{i}.out", "slurm_scripts/run.sh", "--foldNo", f"{i}" ]

        arguments.extend(sys.argv[1:])
        print(f"Submit fold {i} with arguments:\n {arguments}")
        subprocess.run(arguments)


if __name__ == '__main__':
    args = parse_input()
    main(args.pathOutput)