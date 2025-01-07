import optuna
from optuna import create_study


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