import math
import os
import tempfile
import time
import warnings

import cloudpickle
import pandas as pd
from optuna.samplers import TPESampler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, LabelEncoder

os.environ['MPLCONFIGDIR'] = tempfile.mkdtemp()
from pathlib import Path
from statistics import mean

import joblib
import optuna
from dask.distributed import Client
from dask_jobqueue.slurm import SLURMRunner
import dask
from distributed import worker_client, performance_report, get_task_stream
from sklearn.metrics import matthews_corrcoef, r2_score, d2_absolute_error_score
from sklearn.model_selection import StratifiedKFold

from helper.SlurmMemRunner import SLURMMemRunner
from helper.feature_selection import get_selection_HPO_space, get_feature_selection, precompute_feature_selection, \
    impute_and_scale, transform_dict_to_var_dict
from helper.input_parser import parse_input
from helper.load_dataset import generate_xy_split, get_dataset, get_flat_models, is_task_classification, \
    load_feature_groups
from helper.model_training import get_model_HPO_space, get_model
from helper.optuna_helper import copyStudy, categorical_distance_function


# 1. Train/Test Split
# 2. 10-CV
#     - Methode für get-HPO (Model + Selection --> Optuna + static-fold Cross-Val) auf 9-split
#     - Preproc + Trainieren Model auf 9-split mit HPs --> ggf. mit frozen_trial --> alle relevanten Metriken speichern
# 3. Modell mit selbigen wie in CV trainieren auf ganzen Train --> Speichern + HPs
# 4. weitere Eval auf Model --> Speichern

# als separate Main: Multi-Objective für RQ2b

def eval_model_performance(model_instance, X_train_test, precomputed, is_classification):
    y_test = precomputed["y_test"].get().result()
    X_train, X_test = X_train_test
    y_pred = model_instance.predict(X_test)
    if is_classification:
        return matthews_corrcoef(y_test, y_pred)
    else:
        return d2_absolute_error_score(y_test, y_pred)

def train_model(model, is_classification, cores, model_config, X_train_test, precomputed):
    y_train = precomputed["y_train"].get().result()
    X_train, X_test = X_train_test
    model_instance = get_model(model, is_classification, cores, model_config )
    model_instance.fit(X_train, y_train)
    return model_instance

def do_feature_selection(model, is_classification, model_config, precomputed, features, selector_config, feature_groups, cores, verbose = False):
    model_instance_selector = get_model(model, is_classification, 1, model_config )
    return get_feature_selection(precomputed, features, is_classification, selector_config, model_instance_selector, feature_groups, parallelism=cores, verbose=verbose)

def compute_fold(client, model, features, is_classification, model_config, selector_config, feature_groups, precomputed, cores : int, verbose = False)  -> float:
    # feature selection + model training
    X_train_test = client.submit(do_feature_selection, model, is_classification, model_config, precomputed, features, selector_config, feature_groups, cores, verbose,pure=False)
    model_instance = client.submit(train_model, model, is_classification, cores, model_config, X_train_test, precomputed, pure=False)
    return client.submit(eval_model_performance, model_instance, X_train_test, precomputed, is_classification, pure=False)

def objective(trial: optuna.Trial, folds, features, model, should_modelHPO, is_classification, feature_groups, feature_count, cores : int) -> float:
    model_config = get_model_HPO_space(model, trial, is_classification) if should_modelHPO else None
    selector_config = get_selection_HPO_space(features, trial, is_classification, feature_groups, feature_count)
    with worker_client() as client:
        futures = [
            compute_fold(client, model, features, is_classification, model_config, selector_config, feature_groups, future_precompute, cores)
            for i, future_precompute in folds.items()]
        results = client.gather(futures, direct=True)
    return mean(results)

def main(pathData: str, pathOutput: str, features: str, task: str, model: str, modelHPO: bool, selectorHPO: bool, hpo_its: int, foldNo : int):
    Path(pathOutput).mkdir(parents=True, exist_ok=True)
    run_config = {
        "name": f"{task}#{features}#{model}#{modelHPO}#{selectorHPO}#{hpo_its}#{foldNo}",
        "path_data": pathData,
        "path_output": pathOutput,
        "features": features,
        "task": task,
        "model": model,
        "modelHPO": modelHPO,
        "selectorHPO" : selectorHPO,
        "hpo_its": hpo_its,
        "foldNo": foldNo,
    }

    scheduler_options = {
        "interface": "ib0",
    }
    worker_options = {
        "local_directory": "$TMPDIR/",
        "nthreads": 1,
        "interface": "ib0",
        "memory_limit": f"{int(os.getenv("SLURM_CPUS_PER_TASK", 2)) * int(os.getenv("SLURM_MEM_PER_CPU", 2000))}MB"
    }
    scheduler_path = Path(os.path.expandvars("$HOME") + "/tmp/scheduler_files")
    scheduler_path.mkdir(parents=True, exist_ok=True)

    cores = int(os.getenv("OMP_NUM_THREADS", "1"))
    with (SLURMMemRunner(scheduler_file=str(scheduler_path)+"/scheduler-{job_id}.json",
                      worker_options=worker_options, scheduler_options=scheduler_options) as runner):
        with Client(runner, direct_to_workers=True) as client:
            with (
                performance_report(filename=run_config["path_output"] + "/" + run_config["name"] + ".html"),
                get_task_stream() as task_stream
            ):
                print(f"Dask dashboard is available at {client.dashboard_link}")
                client.wait_for_workers(runner.n_workers)

                X, y = get_dataset(pathData, task)
                is_classification = is_task_classification(task)
                label_encoder = None
                if is_classification:
                    label_encoder = LabelEncoder()
                    y = label_encoder.fit_transform(y)
                y = pd.to_numeric(y, downcast='float')
                X_train, X_test, y_train, y_test = generate_xy_split(X, y, pathData+"/folds.txt", foldNo)

                feature_count = X_train.shape[1]

                kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
                model_flatness = get_flat_models(X_train)
                flatness_future = client.scatter(model_flatness, direct=True)

                splits = kf.split(X_train, model_flatness)

                # dask export for better cluster behaviour

                folds = {}
                for i, (train_index, test_index) in enumerate(splits):
                    X_train_inner = X_train.iloc[train_index]
                    X_test_inner = X_train.iloc[test_index]
                    y_train_inner = y_train.iloc[train_index]
                    y_test_inner = y_train.iloc[test_index]
                    # feature preprocessing
                    future_pre = client.submit(precompute_feature_selection,features, is_classification, X_train_inner, X_test_inner, y_train_inner, y_test_inner, flatness_future, 0.9, cores, pure=True)
                    future_pre = client.submit(transform_dict_to_var_dict, future_pre)
                    folds[i] = future_pre

                feature_groups = load_feature_groups(pathData)

                best_params = {}
                journal_path = "no journal"
                verbose = False
                if selectorHPO:
                    objective_function = lambda trial: objective(trial, folds, features, model, modelHPO, is_classification, feature_groups, feature_count, cores)

                    journal_path = run_config["path_output"] + "/" + run_config["name"] + ".journal"
                    journal = optuna.storages.JournalStorage(optuna.storages.journal.JournalFileBackend(journal_path))
                    storage = optuna.integration.dask.DaskStorage(journal)
                    sampler = TPESampler(seed=None, multivariate=True, group=True, constant_liar=True, categorical_distance_func=categorical_distance_function())
                    study = optuna.create_study(storage=storage, direction="maximize", sampler=sampler)

                    if int(os.getenv("SLURM_NTASKS", 1)) < 27:
                        raise ValueError("Not enough worker, needs more than 32")
                    n_jobs = 25 #2 less than tasks for scheduler and main-node
                    n_trials = math.ceil(hpo_its / n_jobs)
                    futures = [
                        client.submit(study.optimize, objective_function, n_trials, pure=False) for _ in range(n_jobs)
                    ]

                    dask.distributed.wait(futures)

                    # train complete model with HPO values
                    best_params = study.best_params
                    frozen_best_trial = study.best_trial


                    # feature selection + model training
                    model_config = get_model_HPO_space(model, frozen_best_trial, is_classification) if modelHPO else None
                    selector_config = get_selection_HPO_space(features, frozen_best_trial, is_classification, feature_groups, X_train.shape[1])
                else:
                    model_config = {}
                    selector_config = {}
                    verbose=True

                model_instance_selector = get_model(model, is_classification, 1, model_config)
                start_FS = time.time()
                precomputed = precompute_feature_selection(features, is_classification, X_train, X_test, y_train, y_test, model_flatness, parallelism=cores)
                precomputed = client.submit(transform_dict_to_var_dict, precomputed)
                fs_future = client.submit(get_feature_selection, precomputed.result(), features, is_classification, selector_config, model_instance_selector, feature_groups, parallelism=cores, verbose=verbose, pure=False)
                X_train, X_test = fs_future.result()
                end_FS = time.time()
                model_instance = get_model(model, is_classification, cores, model_config)
                start_Model =time.time()
                model_instance.fit(X_train, y_train)
                end_Model =time.time()
                model_complete = model_instance

                # export for later use
                output = {
                    "model": model_complete,
                    "X_test": X_test,
                    "y_test": y_test,
                    "label_encoder" : label_encoder,
                    "best_params": best_params,
                    "run_config": run_config,
                    "journal_path": journal_path,
                    "time_Feature" : end_FS - start_FS,
                    "time_Model" : end_Model - start_Model,
                }
            output["task_stream"] = task_stream.data
            path = run_config["path_output"] + "/" + run_config["name"] + ".pkl"
            with open(path, "wb") as f:
                cloudpickle.dump(output, f)
            print(f"Exported model at {path}")


if __name__ == '__main__':
    args = parse_input()
    warnings.filterwarnings("ignore", message="'force_all_finite'")
    main(os.environ.get("HOME")+"/"+os.path.expandvars(args.pathData), os.environ.get("HOME")+"/"+os.path.expandvars(args.pathOutput), args.features, args.task, args.model, args.modelHPO, args.selectorHPO, args.HPOits, args.foldNo)
