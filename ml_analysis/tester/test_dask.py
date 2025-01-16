import math
import os

import dask
import optuna
import pandas as pd
from dask.distributed import Client
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from generate_fold_model import objective
from helper.load_dataset import get_dataset, is_task_classification, generate_xy_split, get_flat_models, \
    load_feature_groups
from model_analyser import run_config

client = Client(n_workers=4, threads_per_worker=1)

pathData = ""
features = ""
task = ""
model = ""
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

# dask export for better cluster behaviour
X_train_dask = dask.delayed(X_train, pure=True)
y_train_dask = dask.delayed(y_train, pure=True)

folds = {
    i: (dask.delayed(train_index, pure=True), dask.delayed(test_index, pure=True)) for i, (train_index, test_index) in enumerate(splits)
}

feature_groups = load_feature_groups(pathData)
feature_groups = dask.delayed(feature_groups)
objective_function = lambda trial: objective(trial, X_train_dask, y_train_dask, folds, features, model, modelHPO, is_classification, feature_groups)

journal_path = run_config["path_output"] + "/" + run_config["name"] + ".journal"
journal = optuna.storages.JournalStorage(optuna.storages.journal.JournalFileBackend(journal_path))
storage = optuna.integration.dask.DaskStorage(journal)
study = optuna.create_study(storage=storage, direction="maximize")

n_jobs = math.ceil((int(os.getenv("SLURM_NTASKS", 7)) - 2) / 3) #2 less than tasks for scheduler and main-node
n_trials = math.ceil(hpo_its / n_jobs)
futures = [
    client.submit(study.optimize, objective_function, n_trials, pure=False) for _ in range(n_jobs)
]

dask.distributed.wait(futures)