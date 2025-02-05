from typing import Any

from optuna import Trial
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, AdaBoostClassifier, AdaBoostRegressor
import xgboost as xgb
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC, SVR


def get_model(model : str, isClassification : bool, parallelism : int = 1, model_args=None):
    if model_args is None:
        model_args = {}
    if "hidden_layer_sizes" in model_args:
        model_args["hidden_layer_sizes"] = [int(x) for x in model_args["hidden_layer_sizes"].split("#")]
    model_args["n_jobs"] = parallelism
    model_args["random_state"] = 42
    match model:
        case "randomForest":
            return RandomForestClassifier(**model_args) if isClassification else  RandomForestRegressor(**model_args)
        case "gradboostForest":
            return xgb.XGBClassifier(**model_args) if isClassification else xgb.XGBRegressor(**model_args)
        case "SVM":
            del model_args["n_jobs"]
            if not isClassification :
                del model_args["random_state"]
            return SVC(**model_args) if isClassification else SVR(**model_args)
        case "kNN":
            del model_args["random_state"]
            return KNeighborsClassifier(**model_args) if isClassification else KNeighborsRegressor(**model_args)
        case "adaboost":
            del model_args["n_jobs"]
            return AdaBoostClassifier(**model_args) if isClassification else AdaBoostRegressor(**model_args)
        case "MLP":
            del model_args["n_jobs"]
            return MLPClassifier(**model_args) if isClassification else MLPRegressor(**model_args)
        case _:
            raise ValueError("Unknown model")

def get_model_HPO_space(model : str, trial : Trial, isClassification : bool) -> dict[str, Any]:
    match model:
        case "randomForest":
            return {
                "n_estimators" : trial.suggest_int("n_estimators", 10, 1000),
                "max_depth" : trial.suggest_categorical("max_depth", [None, 10, 50, 100, 500, 1000]),
            }
        case "gradboostForest":
            return {
                "n_estimators": trial.suggest_int("n_estimators", 100, 10000),
                "max_depth": trial.suggest_int("max_depth", 3, 25),
                "min_child_weight" : trial.suggest_float("min_child_weight", .8, 5),
                "subsample" : trial.suggest_float("subsample", .2, 1)
            }
        case "SVM":
             params = {
                "kernel" : trial.suggest_categorical("kernel", ["linear", "rbf", "sigmoid", "poly"]),
                "C" : trial.suggest_float("C", 10e-5, 10e5, log=True),
                "gamma" : trial.suggest_float("gamma", 10e-5, 10e5, log=True),
                "cache_size" : 750, # use mem better
                "max_iter": 10000 #set limit to limit max runtime
             }
             if params["kernel"] == "poly":
                 params["degree"] = trial.suggest_int("degree", 1, 5)
             if not isClassification:
                 params["epsilon"] = trial.suggest_float("epsilon", .001, 1)
             return params
        case "kNN":
            return {
                "n_neighbors" : trial.suggest_int("n_neighbors", 3, 50),
                "weights" : trial.suggest_categorical("weights", ["uniform", "distance"]),
                "p" : trial.suggest_float("p", .5, 5),
            }
        case "adaboost":
            return {
                "n_estimators" : trial.suggest_int("n_estimators", 50, 300),
                "learning_rate" : trial.suggest_float("learning_rate", .001, 10),
            }
        case "MLP":
            return {
                "hidden_layer_sizes" : trial.suggest_categorical("hidden_layer_sizes", ["100", "100#50#10#50", "100#50#10", "20#20#20", "50#10#50", "100#25#11#7#5#3"]),
                "activation" : trial.suggest_categorical("activation", ["relu", "tanh", "logistic", "identity"]),
                "alpha" : trial.suggest_float("alpha", .0001, .05),
            }
        case _:
            raise ValueError("Unknown model")

