import itertools

from generate_fold_model import create_run_name
from helper.input_parser import get_feature_list, get_model_list, get_task_list

def is_modelHPO(feature: str) -> bool:
    return is_selectorHPO(feature)

def is_selectorHPO(feature: str) -> bool:
    return feature not in ["genetic"]

def get_task_count(feature: str) -> int:
    if feature in ["genetic"]:
        return 256
    else:
        return 256

def get_runtime() -> str:
    #todo add times
    pass

def check_valid(feature: str, model: str, task: str) -> bool:
    if feature == "harris-hawks":
        return False
    if feature == "RFE" and not (model == "gradboostForest" or model == "randomForest" or model == "adaboost"):
        return False
    return True

hpoIts=200
combinations = [ (fold, feature, model, task) for fold, feature, model, task in itertools.product(range(10), get_feature_list(), get_model_list(), get_task_list()) ]

experiments = [
    f"{i} {create_run_name(feature, task, model, is_modelHPO(feature), is_selectorHPO(feature), hpoIts, fold)} {get_task_count(feature)} {get_runtime()} {fold} {feature} {task} {model} {hpoIts} {is_modelHPO(feature)} {is_selectorHPO(feature)}"
    for i, (fold, feature, model, task) in enumerate(combinations) if check_valid(feature, model, task)
]



with open("config.txt", "w") as f:
    f.write("NO NAME TASK_COUNT RUNTIME FOLD_NO FEATURE TASK MODEL HPO_ITS MODEL_HPO SELECTOR_HPO\n")
    f.write("\n".join(experiments))

