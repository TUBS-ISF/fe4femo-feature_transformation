import math
import os

import dask
import optuna
import pandas as pd
from dask import delayed
from dask.distributed import Client
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from ml_analysis.generate_fold_model import objective
from ml_analysis.helper.feature_selection import precompute_feature_selection, impute_and_scale, transform_dict_to_var_dict
from ml_analysis.helper.load_dataset import get_dataset, is_task_classification, generate_xy_split, get_flat_models, \
    load_feature_groups, load_feature_group_times

if __name__ == "__main__":
    client = Client(n_workers=10, threads_per_worker=1)

    pathData = "/mnt/e/Uni/Thesis/fe4femo-feature_transformation/data"
    features = "multisurf"
    task = "runtime_backbone"
    model = "randomForest"
    modelHPO = False
    hpo_its =10
    foldNo =0

    print(f"Dask dashboard is available at {client.dashboard_link}")

    X, y = get_dataset(pathData, task)
    is_classification = is_task_classification(task)
    label_encoder = None
    if is_classification:
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
        y = pd.Series(y)
    X_train, X_test, y_train, y_test = generate_xy_split(X, y, pathData+"/folds.txt", foldNo)
    model_flatness = get_flat_models(X_train)
    flatness_future = client.scatter(model_flatness, direct=True)
    kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    splits = kf.split(X_train, get_flat_models(X_train))


    folds = {}
    for i, (train_index, test_index) in enumerate(splits):
        #X_train_inner = client.scatter(X_train.iloc[train_index])
        #X_test_inner = client.scatter(X_train.iloc[test_index])
        #y_train_inner = client.scatter(y_train.iloc[train_index])
        #y_test_inner = client.scatter(y_train.iloc[test_index])
        # feature preprocessing
        #X_traintest_inner = client.submit(impute_and_scale, X_train_inner, X_test_inner)
        for i, (train_index, test_index) in enumerate(splits):
                        X_train_inner = X_train.iloc[train_index]
                        X_test_inner = X_train.iloc[test_index]
                        y_train_inner = y_train.iloc[train_index]
                        y_test_inner = y_train.iloc[test_index]
        future_pre = client.submit(precompute_feature_selection, features, is_classification,  X_train_inner,        
        X_test_inner,           
        y_train_inner,          
        y_test_inner, model_flatness, 0.9, 8)
        future_pre = client.submit(transform_dict_to_var_dict, future_pre)
        folds[i] = future_pre
 
    cores = int(os.getenv("OMP_NUM_THREADS", "1"))  # or just set cores = 8
    multi_objective = False
    feature_group_times = load_feature_group_times(pathData)

    # Optional but consistent with main pipeline:
    feature_group_times_future = client.scatter(feature_group_times, direct=True)
    feature_groups = load_feature_groups(pathData)
    feature_groups = client.scatter(feature_groups)
    objective_function = lambda trial: objective(trial, folds, features, model, modelHPO, is_classification, feature_groups, feature_group_times_future, X_train.shape[1], cores, multi_objective)

    journal_path = "/tmp/test.journal"
    if os.path.exists(journal_path):
        os.remove(journal_path)
    journal = optuna.storages.JournalStorage(optuna.storages.journal.JournalFileBackend(journal_path))
    storage = optuna.integration.dask.DaskStorage(journal)
    study = optuna.create_study(storage=storage, direction="maximize")

    futures = [
        client.submit(study.optimize, objective_function, n_trials=2, pure=False) for _ in range(1)
    ]
    dask.distributed.wait(futures)

    client.shutdown()