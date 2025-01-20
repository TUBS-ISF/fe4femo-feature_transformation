import itertools
from statistics import mean
from typing import Any

import pandas as pd
from mrmr import mrmr_classif, mrmr_regression
from optuna import Trial
from functools import partial

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import SelectKBest, mutual_info_classif, mutual_info_regression, \
    SequentialFeatureSelector, SelectFromModel, RFECV, VarianceThreshold
from sklearn.model_selection import KFold, cross_val_score
from skrebate import MultiSURF
from zoofs import HarrisHawkOptimization, GeneticOptimization

from external.HFMOEA.main import reduceFeaturesMaxAcc, compute_sol
from external.skfeature.NDFS import ndfs
from external.skfeature.sparse_learning import feature_ranking
from external.svd_entropy import keep_high_contrib_features
from helper.load_dataset import filter_SATzilla, filter_SATfeatPy, filter_FMBA, filter_FMChara


def objective_function_zoo(model, X, y, no_use_X, no_use_y, inner_cv, n_jobs):
    y = y.iloc[:, 0]
    scores = cross_val_score(model, X, y, cv=inner_cv, n_jobs=n_jobs)  # change cv?
    return mean(scores)


def prefilter_features(X_train_in : pd.DataFrame, X_test_in : pd.DataFrame, y_train : pd.Series, threshold : float):
    variance_filter = VarianceThreshold()
    variance_filter.set_output(transform="pandas")
    X_train = variance_filter.fit_transform(X_train_in)
    X_test = variance_filter.transform(X_test_in)

    # https://stackoverflow.com/a/44674459 + own correlation to target
    col_corr = set()
    corr_solution = X_train.corrwith(y_train, axis=0, method="spearman").abs()
    corr_matrix = X_train.corr(method="spearman").abs()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if(corr_matrix.iloc[i, j]) >= threshold:
                if corr_solution.iat[i] > corr_solution.iat[j] and (corr_matrix.columns[i] not in col_corr):
                    col_corr.add(corr_matrix.columns[j])
                if corr_solution.iat[j] > corr_solution.iat[i] and (corr_matrix.columns[j] not in col_corr):
                    col_corr.add(corr_matrix.columns[i])
    to_keep = set(X_train.columns) - set(col_corr)
    return X_train[list(to_keep)], X_test[list(to_keep)]

def precompute_feature_selection(features: str, isClassification : bool, X_train_orig : pd.DataFrame, y_train : pd.Series, X_test_orig : pd.DataFrame, threshold : float = .9, parallelism : int = 1, ):
    if features == "all": # do not prefilter for all
        return {
            "X_train": X_train_orig,
            "X_test": X_test_orig,
        }
    X_train, X_test = prefilter_features(X_train_orig, X_test_orig, y_train, threshold)  # todo leaking?
    ret_dict = {
        "X_train": X_train,
        "X_test": X_test,
    }
    match features:
        case "SATzilla":
            ret_dict["X_train"] = filter_SATzilla(X_train)
            ret_dict["X_test"] = filter_SATzilla(X_test)
        case "SATfeatPy":
            ret_dict["X_train"] = filter_SATfeatPy(X_train)
            ret_dict["X_test"] = filter_SATfeatPy(X_test)
        case "FMBA":
            ret_dict["X_train"] = filter_FMBA(X_train)
            ret_dict["X_test"] = filter_FMBA(X_test)
        case "FM_Chara":
            ret_dict["X_train"] = filter_FMChara(X_train)
            ret_dict["X_test"] = filter_FMChara(X_test)
        case "prefilter":
            pass
        case "kbest-mutalinfo":
            pass
        case "multisurf":
            selector = MultiSURF(n_jobs=parallelism)
            selector.fit(X_train.to_numpy(), y_train.to_numpy())
            ret_dict["top_features"] = selector.top_features_
        case "mRMR":
            pass
        case "RFE":
            pass
        case "harris-hawks":
            pass
        case "genetic":
            pass
        case "HFMOEA":
            ret_dict["sol"] = compute_sol(X_train.to_numpy(), y_train.to_numpy(), isClassification, parallelism)
        case "embedded-tree":
            pass
        case "SVD-entropy":
            pass
        case "NDFS":
            pass
        case "optuna-combined":
            pass
        case _:
            raise ValueError("Invalid Feature Subset")
    return ret_dict


def get_feature_selection(features : str, isClassification : bool, X_train_orig : pd.DataFrame, y_train : pd.Series, X_test_orig : pd.DataFrame, selector_args, estimator, group_dict : dict[str, list[str]], parallelism : int = 1, threshold : float = .9, precomputed = None):
    inner_cv = KFold(n_splits=4, shuffle=True, random_state=42)
    per_estimator_parallel = parallelism // inner_cv.n_splits
    if precomputed is None:
        precomputed = precompute_feature_selection(features=features, isClassification=isClassification, X_train_orig=X_train_orig, y_train=y_train, X_test_orig=X_test_orig, threshold=threshold, parallelism=parallelism)
    max_features = precomputed["X_train"].shape[1]
    match features:
        case "all" | "SATzilla" | "SATfeatPy" | "FMBA" | "FM_Chara" | "prefilter":
            return precomputed["X_train"], precomputed["X_test"]
        case "kbest-mutalinfo":
            X_train = precomputed["X_train"]
            X_test = precomputed["X_test"]
            score_func = partial(mutual_info_classif, random_state=42, n_jobs=parallelism, n_neighbors=selector_args["n_neighbors"] ) if isClassification else partial(mutual_info_regression, random_state=42, n_jobs=parallelism, n_neighbors=selector_args["n_neighbors"])
            selector = SelectKBest(score_func, k=min(max_features, selector_args["k"]))# limit to max feature count after preprocessing
            selector.set_output(transform="pandas")
            selector.fit(X_train, y_train)
            return selector.transform(X_train), selector.transform(X_test)
        case "multisurf":
            X_train = precomputed["X_train"]
            X_test = precomputed["X_test"]
            row_index_to_select = precomputed["top_features"][:min(max_features, selector_args["n_features_to_select"])]# limit to max feature count after preprocessing
            return X_train.iloc[:, row_index_to_select], X_test.iloc[:, row_index_to_select]
        case "mRMR":
            X_train = precomputed["X_train"]
            X_test = precomputed["X_test"]
            selector_args["K"] = min(max_features, selector_args["K"]) # limit to max feature count after preprocessing
            selected_feature_names = mrmr_classif(X_train, y_train, **selector_args, n_jobs=parallelism, show_progress=False) if isClassification else mrmr_regression(X_train, y_train, **selector_args, n_jobs=parallelism, show_progress=False)
            return X_train[selected_feature_names], X_test[selected_feature_names]
        case "RFE":
            X_train = precomputed["X_train"]
            X_test = precomputed["X_test"]
            estimator.set_params(n_jobs=per_estimator_parallel)
            selector_args["min_features_to_select"] = min(max_features, selector_args["min_features_to_select"])  # limit to max feature count after preprocessing
            selector = RFECV(estimator, cv=inner_cv, step=selector_args["step"], min_features_to_select=selector_args["min_features_to_select"], n_jobs=inner_cv.n_splits)
            selector.set_output(transform="pandas")
            selector.fit(X_train, y_train)
            return selector.transform(X_train), selector.transform(X_test)
        case "harris-hawks":
            X_train = precomputed["X_train"]
            X_test = precomputed["X_test"]
            estimator.set_params(n_jobs=per_estimator_parallel)
            X_copy = pd.DataFrame(X_train)
            y_copy = pd.DataFrame(y_train)
            selector = HarrisHawkOptimization(partial(objective_function_zoo, inner_cv=inner_cv, n_jobs=inner_cv.n_splits), **selector_args, minimize=False)
            selected_feature_names = selector.fit(estimator, X_copy, y_copy, X_copy, y_copy, verbose=False)
            return X_train[selected_feature_names], X_test[selected_feature_names]
        case "genetic":
            X_train = precomputed["X_train"]
            X_test = precomputed["X_test"]
            estimator.set_params(n_jobs=per_estimator_parallel)
            X_copy = pd.DataFrame(X_train)
            y_copy = pd.DataFrame(y_train)
            selector = GeneticOptimization(partial(objective_function_zoo, inner_cv=inner_cv, n_jobs=inner_cv.n_splits), **selector_args, minimize=False)
            selected_feature_names = selector.fit(estimator, X_copy, y_copy, X_copy, y_copy, verbose=False)
            return X_train[selected_feature_names], X_test[selected_feature_names]
        case "HFMOEA":
            X_train_np = precomputed["X_train"].to_numpy()
            y_train_np = y_train.to_numpy()
            selector_args["topk"] = min(max_features, selector_args["topk"])  # limit to max feature count after preprocessing
            feature_mask = reduceFeaturesMaxAcc(X_train_np, y_train_np, **selector_args, n_jobs=parallelism, is_classification=isClassification, sol=precomputed["sol"])
            return  precomputed["X_train"].loc[:, feature_mask], precomputed["X_test"].loc[:, feature_mask]
        case "embedded-tree":
            X_train = precomputed["X_train"]
            X_test = precomputed["X_test"]
            forest = RandomForestClassifier(n_estimators=selector_args["e_n_estimators"], max_depth=selector_args["e_max_depth"], n_jobs=parallelism) if isClassification else RandomForestRegressor(n_estimators=selector_args["e_n_estimators"], max_depth=selector_args["e_max_depth"], n_jobs=parallelism)
            selector_args["e_max_features"] = min(max_features, selector_args["e_max_features"])  # limit to max feature count after preprocessing
            model = SelectFromModel(forest, max_features=selector_args["e_max_features"])
            model.set_output(transform="pandas")
            model.fit(X_train, y_train)
            return model.transform(X_train), model.transform(X_test)
        case "SVD-entropy":
            X_train = precomputed["X_train"]
            X_test = precomputed["X_test"]
            boolean_mask = keep_high_contrib_features(X_train)
            return X_train.loc[:, boolean_mask], X_test.loc[:, boolean_mask]
        case "NDFS":
            X_train = precomputed["X_train"]
            X_test = precomputed["X_test"]
            np_view = X_test.to_numpy()
            W = ndfs(np_view, n_clusters=selector_args["n_clusters"], alpha=selector_args["alpha"], beta=selector_args["beta"])
            ranking = feature_ranking(W)
            selector_args["n_features_to_select"] = min(max_features, selector_args["n_features_to_select"])  # limit to max feature count after preprocessing
            sliced = ranking[:selector_args["n_features_to_select"]]
            return X_train.iloc[:, sliced], X_test.iloc[:, sliced]
        case "optuna-combined":
            X_train = precomputed["X_train"]
            X_test = precomputed["X_test"]
            retained_features = set(X_train.columns)
            selected_feature_names_list = [ retained_features & set(v) for k, v in group_dict.items() if selector_args[k]]
            selected_feature_names = list(itertools.chain.from_iterable(selected_feature_names_list))
            return X_train[selected_feature_names], X_test[selected_feature_names]
        case _:
            raise ValueError("Invalid Feature Subset")


def get_selection_HPO_space(features : str, trial : Trial, isClassification : bool, group_dict : dict[str, list[str]], no_features: int) -> dict[str, Any]:
    min_features = min(5, no_features)
    match features:
        case "all" | "SATzilla" | "SATfeatPy" | "FMBA" | "FM_Chara" | "prefilter":
            return {}
        case "kbest-mutalinfo":
            return {
                "k" : trial.suggest_int("k", min_features, no_features),
                "n_neighbors" : trial.suggest_int("n_neighbors", 2, 10)
            }
        case "multisurf":
            return {
                "n_features_to_select" : trial.suggest_int("n_features_to_select", min_features, no_features)
            }
        case "mRMR":
            return {
                "K" : trial.suggest_int("K", min_features, no_features),
                "relevance" : trial.suggest_categorical("relevance", ["f", "ks", "rf"]),
                "denominator" : trial.suggest_categorical("denominator", ["max", "mean"]),
            }
        case "RFE":
            return {
                "step" : trial.suggest_float("step", 0.01, 1),
                "min_features_to_select" : min_features,
            }
        case "genetic":
            return {
                "selective_pressure" : trial.suggest_int("selective_pressure", 1, 5),
                "elitism" : trial.suggest_int("elitism", 1, 5),
                "mutation_rate" : trial.suggest_float("mutation_rate", 0.01, 0.3),
                "population_size" : trial.suggest_int("population_size", 10, 80),
                "n_iteration" : 50 #todo n_iterations?
            }
        case "harris-hawks":
            return {
                "population_size": trial.suggest_int("population_size", 10, 80),
                "beta" : trial.suggest_float("beta", 0.001, 1.999),
                "n_iteration": 50  # todo n_iterations?
            }
        case "HFMOEA":
            return {
                "topk": trial.suggest_int("topk", min_features -1, no_features-1),
                "pop_size" : trial.suggest_int("pop_size", 30, 120),
                "max_gen" : 100,
                "mutation_probability" : trial.suggest_float("mutation_probability", 0.01, 0.3)
            }
        case "embedded-tree":
            return {
                "e_n_estimators": trial.suggest_int("e_n_estimators", 10, 1000),
                "e_max_depth": trial.suggest_categorical("e_max_depth", [10, 50, 100, 500]),
                "e_max_features" : trial.suggest_int("e_max_features", min_features, no_features),
            }
        case "SVD-entropy":
            return {}
        case "NDFS":
            return {
                "alpha" : trial.suggest_float("alpha", 1e-2, 10, log=True),
                "beta" : trial.suggest_float("beta", 1e-2, 10, log=True),
                "n_clusters" : trial.suggest_int("n_clusters", 4, 100),
                "n_features_to_select": trial.suggest_int("n_features_to_select", min_features, no_features),
            }
        case "optuna-combined":
            return {
                group_name : trial.suggest_categorical(group_name, [True, False]) for group_name in group_dict.keys()
            }
        case _:
            raise ValueError("Invalid Feature Subset")
