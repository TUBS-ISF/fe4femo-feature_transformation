import os
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed

from analysis.analysis_helper import get_pickle_dict, list_experiment_instances, ExperimentInstance


def get_task_cumsum(file):
    dictonary = get_pickle_dict(file)
    tasks = []
    for x in dictonary["task_stream"]:
        name = x["key"].split("-")[0]
        time = 0
        for y in x["startstops"]:
            time += y["stop"] - y["start"]
        tasks.append((name, time))

    tasks = pd.DataFrame(tasks, columns=["name", "time"])
    tasks["time"] = pd.to_timedelta(tasks["time"], unit="s")
    cum_sum = tasks.groupby(['name']).sum()
    return cum_sum['time']

def _parallel_wrapper(experiment_instance : ExperimentInstance)->tuple[str, pd.Series]:
    print(f"Handling {experiment_instance}")
    qualities = get_task_cumsum(experiment_instance.path_pickle)
    return experiment_instance.name, qualities

if __name__ == '__main__':
    config_path = Path("~/fe4femo/ml_analysis/slurm_scripts/config.txt").expanduser()
    data_path = Path("~/fe4femo/ml_analysis/out/main/").expanduser()
    out_file = Path("~/fe4femo/ml_analysis/out/task_times/").expanduser()
    out_file.mkdir(parents=True, exist_ok=True)

    experiment_instances = list_experiment_instances(config_path, data_path)
    #ret_gen = Parallel(n_jobs=40, verbose=10, return_as="generator_unordered")(delayed(_parallel_wrapper)(experiment_instance) for experiment_instance in experiment_instances)
    ret_gen = [_parallel_wrapper(experiment_instance) for experiment_instance in experiment_instances]

    for name, series in ret_gen:
        series.to_csv(out_file/ f"{name}.csv")
