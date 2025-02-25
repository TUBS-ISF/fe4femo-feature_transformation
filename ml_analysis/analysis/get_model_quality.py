import os
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed
from pandas import MultiIndex
from sklearn.metrics import matthews_corrcoef, d2_absolute_error_score

from analysis.analysis_helper import get_pickle_dict, list_experiment_instances, ExperimentInstance
from helper.feature_selection import set_njobs_if_possible
from helper.model_training import is_model_classifier


def get_model_quality(file) -> list[float]:
    dictonary = get_pickle_dict(file)
    ret_list = []
    for trial_container in dictonary["trial_container"]:
        model = trial_container.model
        set_njobs_if_possible(model, 1)
        y_pred = model.predict(trial_container.x_test)
        y_test = dictonary["y_test"]
        if is_model_classifier(trial_container.model):
            ret_list.append(matthews_corrcoef(y_test, y_pred))
        else:
            ret_list.append(d2_absolute_error_score(y_test, y_pred).item())
    return ret_list

def _parallel_wrapper(experiment_instance : ExperimentInstance)->tuple[tuple, list[float]]:
    qualities = get_model_quality(experiment_instance.path_pickle)
    index_tuple = experiment_instance.ml_task, experiment_instance.feature_selector, experiment_instance.ml_model, experiment_instance.is_model_hpo, experiment_instance.is_selector_hpo, experiment_instance.is_multi_objective, experiment_instance.fold_no
    return index_tuple, qualities

if __name__ == '__main__':
    config_path = Path("~/fe4femo/ml_analysis/slurm_scripts/config.txt").expanduser()
    data_path = Path("~/fe4femo/ml_analysis/out/main/").expanduser()

    experiment_instances = list_experiment_instances(config_path, data_path)
    #ret_gen = Parallel(n_jobs=os.environ.get("SLURM_CPUS_ON_NODE", -1), verbose=10)(delayed(_parallel_wrapper)(experiment_instance) for experiment_instance in experiment_instances)
    ret_gen = [_parallel_wrapper(experiment_instance) for experiment_instance in experiment_instances]

    index_tuples = []
    values = []
    for index_tuple, qualities in ret_gen:
        index_tuples.append(index_tuple)
        values.append(max(qualities))

    multi_index = MultiIndex.from_tuples(index_tuples, names=["ml_task", "feature_selector", "ml_model", "model_hpo", "selector_hpo", "multi_objective", "fold"])
    df = pd.Series(values, index=multi_index, name="model_quality")
    df.to_csv("model_quality.csv")