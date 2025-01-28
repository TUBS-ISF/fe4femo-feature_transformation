from typing import Callable

import optuna
from optuna import create_study

def distance_e_max_depth(a,b):
    a = int(a)
    b = int(b)
    return abs(a-b)

def distance_max_depth(a,b):
    none_value = 10000000
    if a is None:
        a = none_value
    if b is None:
        b = none_value
    a = int(a)
    b = int(b)
    return abs(a-b)

def categorical_distance_function() -> dict:
    return {
        "e_max_depth" : distance_e_max_depth,
        "max_depth" : distance_max_depth,
    }


def copyStudy(study : optuna.study.Study) -> optuna.study.Study:
    to_study = create_study(
        study_name="copy",
        storage=None,
        directions=study.directions,
        load_if_exists=False,
    )

    for key, value in study._storage.get_study_system_attrs(study._study_id).items():
        to_study._storage.set_study_system_attr(to_study._study_id, key, value)

    for key, value in study.user_attrs.items():
        to_study.set_user_attr(key, value)

    # Trials are deep copied on `add_trials`.
    to_study.add_trials(study.get_trials(deepcopy=False))
    return to_study