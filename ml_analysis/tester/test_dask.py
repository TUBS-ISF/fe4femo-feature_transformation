import math
import os

import dask
import optuna
import pandas as pd
from dask import delayed
from dask.distributed import Client
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from generate_fold_model import objective, impute_and_scale
from helper.feature_selection import precompute_feature_selection
from helper.load_dataset import get_dataset, is_task_classification, generate_xy_split, get_flat_models, \
    load_feature_groups

if __name__ == "__main__":
    client = Client(n_workers=10, threads_per_worker=1)

    pathData = "/home/ubuntu/MA/data"
    features = "multisurf"
    task = "runtime_backbone"
    model = "randomForest"
    modelHPO = True
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

    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    splits = kf.split(X_train, get_flat_models(X_train))


    folds = {}
    for i, (train_index, test_index) in enumerate(splits):
        X_train_inner = client.scatter(X_train.iloc[train_index])
        X_test_inner = client.scatter(X_train.iloc[test_index])
        y_train_inner = client.scatter(y_train.iloc[train_index])
        y_test_inner = client.scatter(y_train.iloc[test_index])
        # feature preprocessing
        X_traintest_inner = client.submit(impute_and_scale, X_train_inner, X_test_inner)
        future_pre = client.submit(precompute_feature_selection, features, is_classification, X_traintest_inner,
                                   y_train_inner, 0.9, 8, pure=True)
        folds[i] = X_traintest_inner, y_train_inner, y_test_inner, future_pre

    feature_groups = load_feature_groups(pathData)
    feature_groups = client.scatter(feature_groups)
    objective_function = lambda trial: objective(trial, folds, features, model, modelHPO, is_classification, feature_groups, X_train.shape[1])

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