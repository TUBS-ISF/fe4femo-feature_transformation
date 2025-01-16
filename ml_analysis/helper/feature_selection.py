import itertools
from statistics import mean
from typing import Any

import pandas as pd
from mrmr import mrmr_classif, mrmr_regression
from optuna import Trial
from functools import partial

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import SelectKBest, mutual_info_classif, mutual_info_regression, \
    SequentialFeatureSelector, SelectFromModel, RFECV
from sklearn.model_selection import KFold, cross_val_score
from skrebate import MultiSURF
from zoofs import HarrisHawkOptimization, GeneticOptimization

from external.HFMOEA.main import reduceFeaturesMaxAcc
from external.skfeature.NDFS import ndfs
from external.skfeature.sparse_learning import feature_ranking
from external.svd_entropy import keep_high_contrib_features
from helper.load_dataset import filter_SATzilla, filter_SATfeatPy, filter_FMBA, filter_FMChara


def objective_function_zoo(model, X, y, no_use_X, no_use_y, inner_cv, n_jobs):
    y = y.iloc[:, 0]
    scores = cross_val_score(model, X, y, cv=inner_cv, n_jobs=n_jobs)  # change cv?
    return mean(scores)

def get_feature_selection(features : str, isClassification : bool, X_train : pd.DataFrame, y_train : pd.Series, X_test : pd.DataFrame, selector_args, estimator, group_dict : dict[str, list[str]], parallelism : int = 1, ):
    inner_cv = KFold(n_splits=4, shuffle=True, random_state=42)
    per_estimator_parallel = parallelism // inner_cv.n_splits
    match features:
        case "all":
            return X_train, X_test
        case "SATzilla":
            return filter_SATzilla(X_train), filter_SATzilla(X_test)
        case "SATfeatPy":
            return filter_SATfeatPy(X_train), filter_SATfeatPy(X_test)
        case "FMBA":
            return filter_FMBA(X_train), filter_FMBA(X_test)
        case "FM_Chara":
            return filter_FMChara(X_train), filter_FMChara(X_test)
        case "kbest-mutalinfo":
            score_func = partial(mutual_info_classif, random_state=42, n_jobs=parallelism, n_neighbors=selector_args["n_neighbors"] ) if isClassification else partial(mutual_info_regression, random_state=42, n_jobs=parallelism, n_neighbors=selector_args["n_neighbors"])
            selector = SelectKBest(score_func, k=selector_args["k"])
            selector.set_output(transform="pandas")
            selector.fit(X_train, y_train)
            return selector.transform(X_train), selector.transform(X_test)
        case "multisurf":
            selector = MultiSURF(**selector_args, n_jobs=parallelism)
            selector.fit(X_train.to_numpy(), y_train.to_numpy())
            row_index_to_select = selector.top_features_[:selector.n_features_to_select]
            return X_train.iloc[:, row_index_to_select], X_test.iloc[:, row_index_to_select]
        case "mRMR":
            selected_feature_names = mrmr_classif(X_train, y_train, **selector_args, n_jobs=parallelism, show_progress=False) if isClassification else mrmr_regression(X_train, y_train, **selector_args, n_jobs=parallelism, show_progress=False)
            return X_train[selected_feature_names], X_test[selected_feature_names]
        case "RFE":
            estimator.set_params(n_jobs=per_estimator_parallel)
            selector = RFECV(estimator, cv=inner_cv, step=selector_args["step"], min_features_to_select=selector_args["min_features_to_select"], n_jobs=inner_cv.n_splits)
            selector.set_output(transform="pandas")
            selector.fit(X_train, y_train)
            return selector.transform(X_train), selector.transform(X_test)
        case "harris-hawks":
            estimator.set_params(n_jobs=per_estimator_parallel)
            X_copy = pd.DataFrame(X_train)
            y_copy = pd.DataFrame(y_train)
            selector = HarrisHawkOptimization(partial(objective_function_zoo, inner_cv=inner_cv, n_jobs=inner_cv.n_splits), **selector_args, minimize=False)
            selected_feature_names = selector.fit(estimator, X_copy, y_copy, X_copy, y_copy, verbose=False)
            return X_train[selected_feature_names], X_test[selected_feature_names]
        case "genetic":
            estimator.set_params(n_jobs=per_estimator_parallel)
            X_copy = pd.DataFrame(X_train)
            y_copy = pd.DataFrame(y_train)
            selector = GeneticOptimization(partial(objective_function_zoo, inner_cv=inner_cv, n_jobs=inner_cv.n_splits), **selector_args, minimize=False)
            selected_feature_names = selector.fit(estimator, X_copy, y_copy, X_copy, y_copy, verbose=False)
            return X_train[selected_feature_names], X_test[selected_feature_names]
        case "HFMOEA":
            feature_mask = reduceFeaturesMaxAcc(X_train, y_train, **selector_args, n_jobs=parallelism, is_classification=isClassification)
            return  X_train.loc[:, feature_mask], X_test.loc[:, feature_mask]
        case "embedded-tree":
            forest = RandomForestClassifier(n_estimators=selector_args["e_n_estimators"], max_depth=selector_args["e_max_depth"], n_jobs=parallelism) if isClassification else RandomForestRegressor(n_estimators=selector_args["e_n_estimators"], max_depth=selector_args["e_max_depth"], n_jobs=parallelism)
            model = SelectFromModel(forest, max_features=selector_args["e_max_features"])
            model.set_output(transform="pandas")
            model.fit(X_train, y_train)
            return model.transform(X_train), model.transform(X_test)
        case "SVD-entropy":
            boolean_mask = keep_high_contrib_features(X_train)
            return X_train.loc[:, boolean_mask], X_test.loc[:, boolean_mask]
        case "NDFS":
            np_view = X_test.to_numpy()
            W = ndfs(np_view, n_clusters=selector_args["n_clusters"], alpha=selector_args["alpha"], beta=selector_args["beta"])
            ranking = feature_ranking(W)
            sliced = ranking[:selector_args["n_features_to_select"]]
            return X_train.iloc[:, sliced], X_test.iloc[:, sliced]
        case "optuna-combined":
            selected_feature_names_list = [v for k, v in group_dict.items() if selector_args[k]]
            selected_feature_names = list(itertools.chain.from_iterable(selected_feature_names_list))
            return X_train[selected_feature_names], X_test[selected_feature_names]


def get_selection_HPO_space(features : str, trial : Trial, isClassification : bool, group_dict : dict[str, list[str]], no_features: int) -> dict[str, Any]:
    min_features = min(5, no_features)
    match features:
        case "all" | "SATzilla" | "SATfeatPy" | "FMBA" | "FM_Chara":
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
