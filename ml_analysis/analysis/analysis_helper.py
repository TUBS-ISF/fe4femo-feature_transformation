from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from cloudpickle import cloudpickle
from optuna import load_study
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend


def get_pickle_dict(file):
    with open(file, "br") as f:
        return cloudpickle.load(f)

def get_optuna_study(file, study_name):
    journal = JournalStorage(JournalFileBackend(file))
    study = load_study(storage=journal, study_name=study_name)
    return study, journal

@dataclass(frozen=True, eq=True)
class ExperimentInstance:
    path_log : Path
    path_pickle : Path
    path_journal : Path
    path_dask : Path
    name: str
    task_count: int
    runtime_limit: int
    fold_no: int
    feature_selector: str
    ml_task: str
    ml_model: str
    hpo_iterations: int
    is_model_hpo: bool
    is_selector_hpo: bool
    is_multi_objective: bool

def list_experiment_instances(config_path: Path, data_path: Path) -> Iterable[ExperimentInstance]:
    with open(config_path, "r") as f:
        next(f)
        for line in f:
            line_array = line.strip().split(" ")
            folds = range(10) if line_array[4] == "-1" else int(line_array[4])
            name = line_array[1]
            for fold in folds:
                yield ExperimentInstance(
                    name=name,
                    task_count=int(line_array[2]),
                    runtime_limit=int(line_array[3]),
                    fold_no=fold,
                    feature_selector=line_array[5],
                    ml_task=line_array[6],
                    ml_model=line_array[7],
                    hpo_iterations=int(line_array[8]),
                    is_model_hpo=line_array[9]  == "True",
                    is_selector_hpo=line_array[10] == "True",
                    is_multi_objective=line_array[11] == "True",

                    path_log= data_path.joinpath(f"{name}.out") if line_array[4] == "-1" else data_path.joinpath(f"{name}#{fold}.out"),
                    path_dask=data_path.joinpath(f"{name}#{fold}.html"),
                    path_pickle=data_path.joinpath(f"{name}#{fold}.pkl"),
                    path_journal=data_path.joinpath(f"{name}#{fold}.journal"),
                )
