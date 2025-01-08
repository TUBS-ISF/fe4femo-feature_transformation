import math
import os
import tempfile

import cloudpickle
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler

os.environ['MPLCONFIGDIR'] = tempfile.mkdtemp()
from pathlib import Path
from statistics import mean

import joblib
import optuna
from dask.distributed import Client
from dask_jobqueue.slurm import SLURMRunner
import dask
from distributed import worker_client, performance_report
from sklearn.metrics import matthews_corrcoef, r2_score, d2_absolute_error_score
from sklearn.model_selection import StratifiedKFold

from helper.SlurmMemRunner import SLURMMemRunner
from helper.feature_selection import get_selection_HPO_space, get_feature_selection
from helper.input_parser import parse_input
from helper.load_dataset import generate_xy_split, get_dataset, get_flat_models, is_task_classification, \
    load_feature_groups
from helper.model_training import get_model_HPO_space, get_model
from helper.optuna_helper import copyStudy


# 1. Train/Test Split
# 2. 10-CV
#     - Methode für get-HPO (Model + Selection --> Optuna + static-fold Cross-Val) auf 9-split
#     - Preproc + Trainieren Model auf 9-split mit HPs --> ggf. mit frozen_trial --> alle relevanten Metriken speichern
# 3. Modell mit selbigen wie in CV trainieren auf ganzen Train --> Speichern + HPs
# 4. weitere Eval auf Model --> Speichern

# als separate Main: Multi-Objective für RQ2b

def impute_and_scale(X_train, X_test):
    imputer = SimpleImputer(keep_empty_features=True, missing_values=pd.NA)
    scaler = RobustScaler()

    X_train = imputer.fit_transform(X_train)
    X_train = scaler.fit_transform(X_train)

    X_test = imputer.transform(X_test)
    X_test = scaler.transform(X_test)

    return X_train, X_test


def compute_fold(dask_X, dask_y, dask_train_index, dask_test_index, model, features, is_classification, dask_model_config, dask_selector_config, dask_feature_groups)  -> float:
    train_index = dask.compute(dask_train_index, traverse=False)[0]
    test_index = dask.compute(dask_test_index, traverse=False)[0]
    X = dask.compute(dask_X, traverse=False)[0]
    y = dask.compute(dask_y, traverse=False)[0]
    model_config = dask.compute(dask_model_config, traverse=False)[0]
    selector_config = dask.compute(dask_selector_config, traverse=False)[0]
    feature_groups = dask.compute(dask_feature_groups, traverse=False)[0]
    X_train = X.iloc[train_index]
    X_test = X.iloc[test_index]
    y_train = y.iloc[train_index]
    y_test = y.iloc[test_index]

    # feature preprocessing
    X_train, X_test = impute_and_scale(X_train, X_test)

    # feature selection + model training
    model_instance_selector = get_model(model, is_classification, 1, model_config )
    X_train, X_test = get_feature_selection(features, is_classification, X_train, y_train, X_test, selector_config, model_instance_selector, feature_groups)
    model_instance = get_model(model, is_classification, 1, model_config )
    model_instance.fit(X_train, y_train)
    y_pred = model_instance.predict(X_test)
    if is_classification:
        return matthews_corrcoef(y_test, y_pred)
    else:
        return d2_absolute_error_score(y_test, y_pred)


def objective(trial: optuna.Trial, dask_X, dask_y, folds, features, model, should_modelHPO, is_classification, dask_feature_groups) -> float:

    feature_groups = dask.compute(dask_feature_groups, traverse=False)[0] if features == "optuna-combined" else None

    model_config = get_model_HPO_space(model, trial, is_classification) if should_modelHPO else None
    selector_config = get_selection_HPO_space(features, trial, is_classification, feature_groups)
    dask_model_config = dask.compute(model_config)[0]
    dask_selector_config = dask.compute(selector_config)[0]

    with worker_client() as client:
        futures = [client.submit(compute_fold, dask_X, dask_y, dask_train_index, dask_test_index, model, features, is_classification, dask_model_config, dask_selector_config, dask_feature_groups) for i, (dask_train_index, dask_test_index) in folds.items() ]
        return mean(client.gather(futures))

def main(pathData: str, pathOutput: str, features: str, task: str, model: str, modelHPO: bool, hpo_its: int, foldNo : int):
    Path(pathOutput).mkdir(parents=True, exist_ok=True)
    run_config = {
        "name": f"{task}_{features}_{model}_{modelHPO}_{hpo_its}_{foldNo}",
        "path_data": pathData,
        "path_output": pathOutput,
        "features": features,
        "task": task,
        "model": model,
        "modelHPO": modelHPO,
        "hpo_its": hpo_its,
        "foldNo": foldNo,
    }

    scheduler_options = {
        "interface": "ib0",
    }
    worker_options = {
        "local_directory": "$TMPDIR/",
        "nthreads": 2,
        "interface": "ib0",
        "memory_limit": f"{int(os.getenv("SLURM_CPUS_PER_TASK", 2)) * int(os.getenv("SLURM_MEM_PER_CPU", 2000))}MB"
    }
    scheduler_path = Path(os.path.expandvars("$HOME") + "/tmp/scheduler_files")
    scheduler_path.mkdir(parents=True, exist_ok=True)
    with (SLURMMemRunner(scheduler_file=str(scheduler_path)+"/scheduler-{job_id}.json",
                      worker_options=worker_options, scheduler_options=scheduler_options) as runner):
        with Client(runner) as client:
            with performance_report(filename=run_config["path_output"] + run_config["name"] + ".html"):
                print(f"Dask dashboard is available at {client.dashboard_link}")
                client.wait_for_workers(runner.n_workers)

                X, y = get_dataset(pathData, task)
                X_train, X_test, y_train, y_test = generate_xy_split(X, y, pathData+"/folds.txt", foldNo)

                kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
                splits = kf.split(X_train, get_flat_models(X_train))

                # dask export for better cluster behaviour
                X_train_dask = dask.delayed(X_train, pure=True)
                y_train_dask = dask.delayed(y_train, pure=True)

                folds = {
                    i: (dask.delayed(train_index, pure=True), dask.delayed(test_index, pure=True)) for i, (train_index, test_index) in enumerate(splits)
                }

                feature_groups = load_feature_groups(pathData)
                feature_groups = dask.delayed(feature_groups)
                is_classification = is_task_classification(task)
                objective_function = lambda trial: objective(trial, X_train_dask, y_train_dask, folds, features, model, modelHPO, is_classification, feature_groups)

                storage = optuna.integration.dask.DaskStorage()
                study = optuna.create_study(storage=storage, direction="maximize")

                n_jobs = math.ceil((int(os.getenv("SLURM_NTASKS", 7)) - 2) / 8) #2 less than tasks for scheduler and main-node
                n_trials = math.ceil(hpo_its / n_jobs)
                futures = [
                    client.submit(study.optimize, objective_function, n_trials, pure=False) for _ in range(n_jobs)
                ]

                dask.distributed.wait(futures)

                # train complete model with HPO values
                best_params = study.best_params
                frozen_best_trial = study.best_trial

                #todo work with best trial
                with joblib.parallel_backend('dask'):
                    # feature preprocessing
                    X_train, X_test = impute_and_scale(X_train, X_test)

                    # feature selection + model training
                    model_config = get_model_HPO_space(model, frozen_best_trial, is_classification) if modelHPO else None
                    selector_config = get_selection_HPO_space(features, frozen_best_trial, is_classification, feature_groups)
                    model_instance_selector = get_model(model, is_classification, 1, model_config)
                    X_train, X_test = get_feature_selection(features, is_classification, X_train, y_train, X_test,
                                                            selector_config, model_instance_selector, feature_groups, parallelism=n_jobs)
                    model_instance = get_model(model, is_classification, n_jobs, model_config)
                    model_instance.fit(X_train, y_train)
                model_complete = model_instance

                # export for later use
                output = {
                    "model": model_complete,
                    "X_test": X_test,
                    "y_test": y_test,
                    "best_params": best_params,
                    "run_config": run_config,
                    "study": copyStudy(study)  #todo investigate if works
                }
                path = run_config["path_output"] + run_config["name"] + ".pkl"
                with open(path, "wb") as f:
                    cloudpickle.dump(output, f)
                print(f"Exported model at {path}")



if __name__ == '__main__':
    args = parse_input()

    main(os.environ.get("HOME")+"/"+os.path.expandvars(args.pathData), os.environ.get("HOME")+"/"+os.path.expandvars(args.pathOutput), args.features, args.task, args.model, args.modelHPO, args.HPOits, args.foldNo)
