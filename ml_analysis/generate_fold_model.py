import os
from pathlib import Path
from statistics import mean

import joblib
import optuna
from dask.distributed import Client
from dask_jobqueue.slurm import SLURMRunner
import dask
from distributed import worker_client
from sklearn.metrics import matthews_corrcoef, r2_score
from sklearn.model_selection import StratifiedKFold

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

def compute_fold(dask_X, dask_y, dask_train_index, dask_test_index, model, features, is_classification, dask_model_config, dask_selector_config, dask_feature_groups)  -> float:
    train_index = dask.compute(dask_train_index, traverse=False)
    test_index = dask.compute(dask_test_index, traverse=False)
    X = dask.compute(dask_X, traverse=False)
    y = dask.compute(dask_y, traverse=False)
    model_config = dask.compute(dask_model_config, traverse=False)
    selector_config = dask.compute(dask_selector_config, traverse=False)
    feature_groups = dask.compute(dask_feature_groups, traverse=False)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    model_instance_selector = get_model(model, is_classification, 1, model_config )
    X_train, X_test = get_feature_selection(features, is_classification, X_train, y_train, X_test, selector_config, model_instance_selector, feature_groups)
    model_instance = get_model(model, is_classification, 1, model_config )
    model_instance.fit(X_train, y_train)
    y_pred = model_instance.predict(X_test)
    if is_classification:
        return matthews_corrcoef(y_test, y_pred)
    else:
        r2 = r2_score(y_test, y_pred)
        n, p = X_test.shape
        return 1 - ((1-r2)* ((n-1)/(n-p-1))) # adjusted R2


def objective(trial: optuna.Trial, dask_X, dask_y, dask_folds, features, model, should_modelHPO, is_classification, dask_feature_groups) -> float:

    folds = dask.compute(dask_folds, traverse=False)

    feature_groups = dask.compute(dask_feature_groups, traverse=False) if features == "optuna-combined" else None

    model_config = get_model_HPO_space(model, trial, is_classification) if should_modelHPO else None
    selector_config = get_selection_HPO_space(features, trial, is_classification, feature_groups)
    dask_model_config = dask.compute(model_config)
    dask_selector_config = dask.compute(selector_config)

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
        "local_directory": "/out/dask_tmp/",
        "nthreads": 4,
        "interface": "ib0",
        "memory_limit": f"{int(os.getenv("SLURM_CPUS_PER_TASK", 2)) * int(os.getenv("SLURM_MEM_PER_CPU", 2000))}MB"
    }
    with (SLURMRunner(scheduler_file=os.path.expandvars("$HOME") + "/tmp/scheduler_files/scheduler-{job_id}.json",
                      worker_options=worker_options, scheduler_options=scheduler_options) as runner):
        with Client(runner) as client:
            print(f"Dask dashboard is available at {client.dashboard_link}")
            client.wait_for_workers(runner.n_workers)

            X, y = get_dataset(pathData, task)
            X_train, X_test, y_train, y_test = generate_xy_split(X, y, pathData+"/folds.txt", foldNo)

            kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
            splits = kf.split(X_train, get_flat_models(X_train))

            # dask export for better cluster behaviour
            X_train = dask.delayed(X_train, pure=True)
            y_train = dask.delayed(y_train, pure=True)

            folds = {
                i: (dask.delayed(train_index, pure=True), dask.delayed(test_index, pure=True)) for i, (train_index, test_index) in enumerate(splits)
            }

            feature_groups = load_feature_groups(pathData)
            feature_groups = dask.delayed(feature_groups)
            is_classification = is_task_classification(task)
            objective_function = lambda trial: objective(trial, X_train, y_train, folds, features, model, modelHPO, is_classification, feature_groups)

            storage = optuna.integration.dask.DaskStorage()
            study = optuna.create_study(storage=storage, direction="maximize")

            n_jobs = int(os.getenv("SLURM_NTASKS", 7)) - 2 #2 less than tasks for scheduler and main-node
            n_trials = (hpo_its / n_jobs) + 1
            futures = [
                client.submit(study.optimize, objective_function, n_trials, pure=False) for _ in range(n_jobs)
            ]

            dask.distributed.wait(futures)

            # train complete model with HPO values
            best_params = study.best_params
            frozen_best_trial = study.best_trial

            #todo work with best trial
            with joblib.parallel_backend('dask'):
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
            pickle_path = joblib.dump(output, run_config["path_output"] + run_config["name"] + ".z", compress=True, protocol=5)
            print(f"Exported model at {pickle_path}")



if __name__ == '__main__':
    args = parse_input()

    main(os.path.expandvars(args.pathData), os.path.expandvars(args.pathOutput), args.features, args.task, args.model, args.modelHPO, args.HPOits, args.foldNo)