import os
import subprocess
from pathlib import Path


#
# 1. 10-CV
#     - Methode für get-HPO (Model + Selection --> Optuna + static-fold Cross-Val) auf 9-split
#     - Preproc + Trainieren Model auf 9-split mit HPs --> ggf. mit frozen_trial --> alle relevanten Metriken speichern
# 3. Modell mit selbigen wie in CV trainieren auf ganzen Train --> Speichern + HPs
# 4. weitere Eval auf Model --> Speichern

# als separate Main: Multi-Objective für RQ2b


if __name__ == '__main__':
    outputdir = "fe4femo/ml_analysis/out/selector_test/"

    i = 0
    features = ["all", "prefilter", "SATzilla", "SATfeatPy", "FMBA", "FM_Chara",
                                 "kbest-mutalinfo", "multisurf", "mRMR", "RFE", "harris-hawks",
                                 "genetic", "HFMOEA", "embedded-tree", "SVD-entropy", "NDFS", "optuna-combined"]
    pathData = "raphael-dunkel-master/data/"
    task = "runtime_backbone"
    model = "randomForest"
    hpo_its = "150"
    home = os.getenv('HOME', "/home/ul/ul_student/ul_ppm61")
    Path(home+"/"+outputdir).mkdir(parents=True, exist_ok=True)
    for feature in features:
        name = f"{task}#{feature}#{model}#True#{hpo_its}#{i}"
        arguments = ["sbatch", "-J", name, "--partition=multiple_il", f"--output={home}/{outputdir}/{name}.out", "../slurm_scripts/run.sh",
                     "--foldNo", f"{i}", "--features", feature, "--task", task, "--model", model, "--modelHPO", "--HPOits", hpo_its, pathData, outputdir ]
        print(f"Submit fold {i} with arguments:\n {arguments}")
        subprocess.run(arguments)