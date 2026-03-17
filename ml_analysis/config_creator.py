import itertools
import math
from datetime import timedelta

from helper.input_parser import (
    get_model_list,
    get_non_heavy_feature_list,
    get_task_list,
    get_transformation_list,
)
from helper.run_naming import create_run_name

#
# Campaign configuration (explicit/hardcoded by design)
#
FIXED_TASK = "runtime_backbone"
INDIVIDUAL_FOLDS = True
INCLUDE_HPO = False
INCLUDE_MULTI_OBJECTIVE = False

# Optional campaign filters. Keep as None to use full defaults.
SELECTED_FEATURES = None
SELECTED_MODELS = None
SELECTED_TRANSFORMATIONS = None

EXCLUDED_FEATURES = {"multisurf", "SVD-entropy"}

INVALID_FEATURE_TRANSFORMATIONS = {
    ("SATzilla", "pca"),
    ("SATzilla", "nystroem-rbf"),
    ("SATzilla", "bin-ordinal"),
    ("SATfeatPy", "pca"),
    ("SATfeatPy", "nystroem-rbf"),
    ("SATfeatPy", "bin-ordinal"),
    ("FM_Chara", "pca"),
    ("FM_Chara", "nystroem-rbf"),
    ("FM_Chara", "bin-ordinal"),
    ("FMBA", "pca"),
    ("FMBA", "nystroem-rbf"),
    ("FMBA", "bin-ordinal"),
}


def is_modelHPO(feature: str) -> bool:
    return INCLUDE_HPO

def is_selectorHPO(feature: str) -> bool:
    return INCLUDE_HPO

def get_task_count(feature: str) -> int:
    return 3

def get_runtime(hpoIts: int, feature: str, individual_folds : bool, multi_objective: bool) -> str:
    selector_modifier_m = {
        "all" : 160,
        "prefilter" : 100,
        "SATzilla" : 32,
        "SATfeatPy" : 40,
        "FMBA" : 18,
        "FM_Chara" : 18,
        "kbest-mutalinfo" : 45,
        "multisurf" : 60,
        "mRMR" : 70,
        "RFE" : 6*60,
        "genetic" : 3*60 + 30,
        "HFMOEA" : 50*60 + 30,
        "embedded-tree" : 175,
        "SVD-entropy" : 30,
        "NDFS" : 36,
        "optuna-combined" : 70
    }

    final_modifier = 1.8
    value = selector_modifier_m[feature]
    if feature != "genetic" and hpoIts > 0:
        value *= hpoIts / 150
    if individual_folds:
        value /= 8 #less since additional startup for container
    if multi_objective:
        value *= 2
    value *= final_modifier
    value += 20

    value = math.ceil(value)
    value = min(value, 72*60)
    return str(value)

def get_HPO_its(multi_objective: bool) -> int:
    if not INCLUDE_HPO:
        return 0
    if multi_objective:
        return 300
    else:
        return 150


def check_valid(feature: str, model: str, task: str, multi_objective: bool, transform: str) -> bool:
    if feature in EXCLUDED_FEATURES:
        return False
    if feature == "RFE" and not (model == "gradboostForest" or model == "randomForest" or model == "adaboost"):
        return False
    if (feature, transform) in INVALID_FEATURE_TRANSFORMATIONS:
        return False
    if multi_objective and feature not in ["optuna-combined"]:
        return False
    return True

#todo configure
def check_desired(feature: str, model: str, task: str, multi_objective: bool) -> bool:
    return True


if __name__ == '__main__':
    if FIXED_TASK not in get_task_list():
        raise ValueError(f"Task '{FIXED_TASK}' not found in get_task_list()")

    selected_features = SELECTED_FEATURES if SELECTED_FEATURES is not None else [
        feature for feature in get_non_heavy_feature_list() if feature not in EXCLUDED_FEATURES
    ]
    selected_models = SELECTED_MODELS if SELECTED_MODELS is not None else get_model_list()
    selected_transformations = SELECTED_TRANSFORMATIONS if SELECTED_TRANSFORMATIONS is not None else get_transformation_list()

    folds = range(10) if INDIVIDUAL_FOLDS else [-1]
    objective_modes = [INCLUDE_MULTI_OBJECTIVE]
    combinations = [
        (fold, feature, model, FIXED_TASK, multi_objective, transform)
        for fold, feature, model, multi_objective, transform in itertools.product(
            folds,
            selected_features,
            selected_models,
            objective_modes,
            selected_transformations,
        )
        if check_valid(feature, model, FIXED_TASK, multi_objective, transform) and check_desired(feature, model, FIXED_TASK, multi_objective)
    ]

    combinations.sort(key=lambda c: int(get_runtime(get_HPO_its(c[4]), c[1], INDIVIDUAL_FOLDS, c[4])))

    experiments = [
        f"{i} {create_run_name(feature, task, model, is_modelHPO(feature), is_selectorHPO(feature), get_HPO_its(multi_objective), multi_objective, fold)}#{transform} {get_task_count(feature)} {get_runtime(get_HPO_its(multi_objective), feature, INDIVIDUAL_FOLDS, multi_objective)} {fold} {feature} {task} {model} {get_HPO_its(multi_objective)} {is_modelHPO(feature)} {is_selectorHPO(feature)} {multi_objective} {transform}"
        for i, (fold, feature, model, task, multi_objective, transform) in enumerate(combinations)
    ]

    runtime_estimation = 0
    for experiment in experiments:
        runtime_estimation += int(experiment.split(" ")[3])
    print(f"Total estimated sequential runtime: {timedelta(minutes=runtime_estimation)}")

    file_count = 0
    for experiment in experiments:
        fold = int(experiment.split(" ")[4])
        uses_hpo = experiment.split(" ")[9] == "True" or experiment.split(" ")[10]  == "True"
        file_count += 1 # for .out file
        tmp_count = 2
        if uses_hpo:
            tmp_count += 1
        if fold == -1:
            tmp_count *= 10
        file_count += tmp_count
    print(f"Total number of generated files: {file_count}")


    with open("config.txt", "w") as f:
        f.write("NO NAME TASK_COUNT RUNTIME FOLD_NO FEATURE TASK MODEL HPO_ITS MODEL_HPO SELECTOR_HPO MULTI_OBJECTIVE TRANSFORMATION\n")
        f.write("\n".join(experiments))

