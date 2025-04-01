from pathlib import Path

import pandas as pd
from pandas import MultiIndex

from analysis.analysis_helper import list_experiment_instances

def get_relevant_indices(feature_selector: str) -> tuple[list[str], list[str]]:
    match feature_selector:
        case "all" | "SATzilla" | "SATfeatPy" | "FMBA" | "FM_Chara" | "prefilter" | "kbest-mutalinfo" | "mRMR" | "RFE" | "embedded-tree"| "NDFS" | "optuna-combined":
            return ['precompute_feature_selection', 'get_feature_selection', 'do_feature_selection'], []
        case "multisurf":
            return ['precompute_feature_selection', 'get_feature_selection', 'do_feature_selection'], ['batch_of_MultiSURF_compute_scores', 'batch_of_find_neighbors']
        case "genetic":
            return ['precompute_feature_selection', 'get_feature_selection', 'compute_cv'], []
        case "HFMOEA":
            return ['precompute_feature_selection', 'get_feature_selection', 'info_gain', 'Dispersion_ratio', 'Fisher_score', 'MAD', 'MI', 'PCC', 'Relief', 'SCC', 'chi_square', 'compute_MI', 'compute_MI_mod', 'compute_cv', 'feature_selection_sim'], []
        case "SVD-entropy":
            return ['precompute_feature_selection', 'get_feature_selection', 'do_feature_selection'], ['batch_of_compute_feature_contribution']
        case _:
            raise ValueError("Unknown feature selector")

def get_sum_if_exists(series: pd.Series, indices: list[str], batch_indices: list[str]):
    val = series.loc[indices].sum()
    for partial_index in batch_indices:
        val += series.loc[lambda x: [partial_index in y for y in x.index]].sum()
    return val


def handle_task_sorting(feature_selector: str, task_times: pd.Series):
    return get_sum_if_exists(task_times, *get_relevant_indices(feature_selector))



def transform_task_times(config_path: Path, data_path: Path, ) -> pd.Series:
    experiment_instances = list_experiment_instances(config_path, Path(".."))
    tmp_list = []
    index_tuples = []
    for experiment_instance in experiment_instances:
        print(f"Handle {experiment_instance.name}#{experiment_instance.fold_no}")
        file = data_path.joinpath(f"{experiment_instance.name}#{experiment_instance.fold_no}.csv")
        series = pd.read_csv(file, index_col=0, parse_dates=[1], date_parser=pd.to_timedelta)['time']
        index_tuple = experiment_instance.ml_task, experiment_instance.feature_selector, experiment_instance.ml_model, experiment_instance.is_model_hpo, experiment_instance.is_selector_hpo, experiment_instance.is_multi_objective, experiment_instance.fold_no
        index_tuples.append(index_tuple)
        tmp_list.append(handle_task_sorting(experiment_instance.feature_selector, series))
    multi_index = MultiIndex.from_tuples(index_tuples,
                                         names=["ml_task", "feature_selector", "ml_model", "model_hpo", "selector_hpo",
                                                "multi_objective", "fold"])
    return pd.Series(tmp_list, index=multi_index, name="task_time")

if __name__ == '__main__':
    config_path = Path("~/fe4femo/ml_analysis/slurm_scripts/config.txt").expanduser()
    data_path = Path("~/fe4femo/ml_analysis/out/task_times/").expanduser()
    out_file = Path("~/fe4femo/ml_analysis/out/task_times.csv").expanduser()

    df = transform_task_times(config_path, data_path)
    print(df)
    df.to_csv(out_file)