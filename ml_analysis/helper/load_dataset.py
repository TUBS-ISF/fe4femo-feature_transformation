import math

import dask
import numpy as np
import pandas as pd
from collections import Counter

from dask.bag.text import delayed
from sklearn.model_selection import train_test_split, StratifiedKFold

sharpsat_names = [
        "approxmc",
        "countantom",
        "d4v2_23",
        "d4v2_24",
        "exactmc_arjun",
        "ganak",
        "sharpsattd"
    ]

suffix_mc = "_modelCount"
suffix_time = "_wallclockTimeS"


def load_dataset(path : str, subpath : str) -> pd.DataFrame:
    df = pd.read_csv(path+ "/" + subpath, header=0, low_memory=False, index_col="modelNo")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df



def load_feature_data(main_path : str) -> pd.DataFrame:
    df = load_dataset(main_path, "featureExtraction/values.csv")
    df['FM_Characterization_1/FM_Characterization/ANALYSIS/Partial_variability/value'] = df['FM_Characterization_1/FM_Characterization/ANALYSIS/Partial_variability/value'].astype(float)
    df['FM_Characterization_1/FM_Characterization/ANALYSIS/Configurations/value'] = df['FM_Characterization_1/FM_Characterization/ANALYSIS/Configurations/value'].astype(float)
    for index, row in df.dtypes[df.dtypes == "object"].items():
        df[index] = df[index].astype("boolean")
    df.columns = df.columns.map(str)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df


def filter_SATzilla(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[:, df.columns.str.startswith('SATZilla')]


def filter_FMBA(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[:, df.columns.str.startswith('FMBA')]


def filter_FMChara(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[:, df.columns.str.startswith('FM_Characterization')]


def filter_SATfeatPy(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[:, df.columns.str.startswith('SATfeatPy')]


def load_sat_runtime(main_path : str) -> pd.DataFrame:
    df = load_dataset(main_path, "runtime/sat.csv")
    return df.loc[:, 'wallclockTimeS'].astype(float)


def load_sat_mem(main_path : str) -> pd.DataFrame:
    df = load_dataset(main_path, "runtime/sat.csv")
    return df.loc[:, 'memUseMB'].astype(float)


def load_backbone_runtime(main_path : str) -> pd.DataFrame:
    df = load_dataset(main_path, "runtime/backbone.csv")
    return df.loc[:, 'wallclockTimeS'].astype(float)


def load_backbone_mem(main_path : str) -> pd.DataFrame:
    df = load_dataset(main_path, "runtime/backbone.csv")
    return df.loc[:, 'memUseMB'].astype(float)


def load_backbone_size(main_path : str) -> pd.DataFrame:
    df = load_dataset(main_path, "runtime/backbone.csv")
    return df.loc[:, 'backboneSize'].astype(int)


def load_spur_runtime(main_path : str, s : int) -> pd.DataFrame:
    df = load_dataset(main_path, f"runtime/spur_{s}.csv")
    return df.loc[:, 'wallclockTimeS'].astype(float)


def load_spur_mem(main_path : str, s : int) -> pd.DataFrame:
    df = load_dataset(main_path, f"runtime/spur_{s}.csv")
    return df.loc[:, 'memUseMB'].astype(float)


def check_ssat_value(row: pd.Series) -> int:
    row_na = row.dropna()
    counts = [int(row[f"{i}{suffix_mc}"]) for i in sharpsat_names if f"{i}{suffix_mc}" in row_na.index]
    if len(counts) == 0:
        return -1
    else:
        return Counter(counts).most_common(1)[0][0]


def check_best_sSAT_solver(row : pd.Series) -> str:
    count = check_ssat_value(row)
    row_na = row.dropna()
    if count == -1:
        time_dict = {solver: row[solver + suffix_time] for solver in sharpsat_names}
    else:
        time_dict = {solver : row[solver+suffix_time] for solver in sharpsat_names if f"{solver}{suffix_mc}" in row_na.index and int(row[solver+suffix_mc]) == count}
    return min(time_dict, key=time_dict.get)


def load_algo_selection(main_path : str) -> pd.DataFrame:
    df = load_dataset(main_path, "runtime/sharpsat.csv")
    return df.apply(check_best_sSAT_solver, axis=1)

def load_value_ssat(main_path : str) -> pd.DataFrame:
    df = load_dataset(main_path, "runtime/sharpsat.csv")
    return df.apply(check_ssat_value, axis=1)

def get_flat_models(df: pd.DataFrame) -> pd.Series:
    return df.apply(lambda row : "flat" if row["FMBA_8/FMBA/TreeDepth"] == 2 else "normal", axis=1)

def get_dataset(path: str, task: str):
    X = load_feature_data(path)
    match task:
        case "runtime_sat":
            y = load_sat_runtime(path)
        case "runtime_backbone":
            y = load_backbone_runtime(path)
        case "runtime_spur":
            y = load_spur_runtime(path, 100)
        case "value_ssat":
            y = load_value_ssat(path)
        case "value_backbone":
            y = load_backbone_size(path)
        case "algo_selection":
            y = load_algo_selection(path)
        case _:
            raise ValueError("Invalid task")
    return X, y

def is_task_classification(task : str) -> bool:
    match task:
        case "runtime_sat":
            return False
        case "runtime_backbone":
            return False
        case "runtime_spur":
            return False
        case "value_ssat":
            return False
        case "value_backbone":
            return False
        case "algo_selection":
            return True
        case _:
            raise ValueError("Invalid task")

def load_feature_groups(path: str) -> dict[str, list[str]]:
    df = pd.read_csv(path+"/featureExtraction/groupMapping.csv", header=0)
    return df.groupby("groupName")['featureName'].apply(list).to_dict

def generate_xy_split(X, y, fold_path, foldNo) -> tuple:
    with open(fold_path) as f:
        splits = [[int(x) for x in line.split()] for line in f]
        train_index = splits[2*foldNo]
        test_index = splits[2*foldNo+1]
        return X.iloc[train_index], X.iloc[test_index], y.iloc[train_index], y.iloc[test_index]



