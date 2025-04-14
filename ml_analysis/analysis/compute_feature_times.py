import os
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed
from pandas import MultiIndex

from analysis.analysis_helper import get_pickle_dict, list_experiment_instances, ExperimentInstance
from helper.load_dataset import load_feature_groups, load_feature_group_times


def get_feature_cumsum(file, feature_groups: dict[str, list[str]], feature_group_times: pd.DataFrame, index_dict : dict[str, str]) -> pd.DataFrame:
    dictonary = get_pickle_dict(file)
    active_features = set(dictonary["trial_container"][0].x_test.columns.values) #todo fix: currently only picks first (no problem for everything except multi-target optimiziation)
    active_groups = [group for group, feature_list in feature_groups.items() if
                     any(feature in active_features for feature in feature_list)]
    df = pd.DataFrame(feature_group_times[active_groups].sum(axis=1).rename('feature_time')).reset_index()
    for name, value in index_dict.items():
        df[name] = value
    return df

def _parallel_wrapper(experiment_instance : ExperimentInstance, feature_groups: dict[str, list[str]], feature_group_times: pd.DataFrame, )-> pd.DataFrame:
    print(f"Handling {experiment_instance}")
    index_dict = {
        "ml_task": experiment_instance.ml_task,
        "feature_selector": experiment_instance.feature_selector,
        "ml_model": experiment_instance.ml_model,
        "model_hpo": experiment_instance.is_model_hpo,
        "selector_hpo": experiment_instance.is_selector_hpo,
        "multi_objective": experiment_instance.is_multi_objective,
        "fold": experiment_instance.fold_no,
    }
    return get_feature_cumsum(experiment_instance.path_pickle, feature_groups, feature_group_times, index_dict)

if __name__ == '__main__':
    config_path = Path("~/fe4femo/ml_analysis/slurm_scripts/config.txt").expanduser()
    data_path = Path("~/fe4femo/ml_analysis/out/main/").expanduser()
    feature_data_path = Path("~/raphael-dunkel-master/data/").expanduser()
    out_file = Path("~/fe4femo/ml_analysis/out/feature_times.csv").expanduser()

    feature_groups = load_feature_groups(str(feature_data_path))
    feature_group_times = load_feature_group_times(str(feature_data_path))

    experiment_instances = list_experiment_instances(config_path, data_path)
    #ret_gen = Parallel(n_jobs=40, verbose=10, return_as="generator_unordered")(delayed(_parallel_wrapper)(experiment_instance, feature_groups, feature_group_times) for experiment_instance in experiment_instances)
    ret_gen = (_parallel_wrapper(experiment_instance, feature_groups, feature_group_times) for experiment_instance in experiment_instances)


    df = pd.concat(ret_gen, axis=0, ignore_index=True)
    df.to_csv(out_file, index=False)
