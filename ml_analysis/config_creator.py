import itertools
import math
from datetime import timedelta

from generate_fold_model import create_run_name
from helper.input_parser import get_feature_list, get_model_list, get_task_list

def is_modelHPO(feature: str) -> bool:
    return is_selectorHPO(feature)

def is_selectorHPO(feature: str) -> bool:
    return feature not in ["genetic", "HFMOEA"]

def get_task_count(feature: str) -> int:
    if feature in ["genetic", "HFMOEA"]:
        return 4
    else:
        return 3

def get_runtime(hpoIts: int, feature: str) -> str:
    selector_modifier_m = {
        "all" : 30,
        "prefilter" : 10,
        "SATzilla" : 5,
        "SATfeatPy" : 8,
        "FMBA" : 5,
        "FM_Chara" : 4,
        "kbest-mutalinfo" : 7,
        "multisurf" : 8,
        "mRMR" : 6,
        "RFE" : 45,
        "genetic" : 90,
        "HFMOEA" : 120,
        "embedded-tree" : 10,
        "SVD-entropy" : 4,
        "NDFS" : 8,
        "optuna-combined" : 6
    }

    final_modifier = 1.2
    value = selector_modifier_m[feature]
    if feature != "genetic":
        value *= hpoIts / 50
    value *= final_modifier

    return str(math.ceil(value))

def check_valid(feature: str, model: str, task: str) -> bool:
    if feature == "harris-hawks":
        return False
    if feature == "RFE" and not (model == "gradboostForest" or model == "randomForest" or model == "adaboost"):
        return False
    return True

def check_desired(feature: str, model: str, task: str):
    return True


if __name__ == '__main__':
    hpoIts=150
    combinations = [
        (fold, feature, model, task) for fold, feature, model, task in itertools.product(range(10), get_feature_list(), get_model_list(), get_task_list())
        if check_valid(feature, model, task) and check_desired(feature, model, task)
    ]

    experiments = [
        f"{i} {create_run_name(feature, task, model, is_modelHPO(feature), is_selectorHPO(feature), hpoIts, fold)} {get_task_count(feature)} {get_runtime(hpoIts, feature)} {fold} {feature} {task} {model} {hpoIts} {is_modelHPO(feature)} {is_selectorHPO(feature)}"
        for i, (fold, feature, model, task) in enumerate(combinations)
    ]

    runtime_estimation = 0
    for experiment in experiments:
        runtime_estimation += int(experiment.split(" ")[3])
    print(f"Total estimated sequential runtime: {timedelta(minutes=runtime_estimation)}")



    with open("config.txt", "w") as f:
        f.write("NO NAME TASK_COUNT RUNTIME FOLD_NO FEATURE TASK MODEL HPO_ITS MODEL_HPO SELECTOR_HPO\n")
        f.write("\n".join(experiments))

