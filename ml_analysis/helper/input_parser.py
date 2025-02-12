import argparse

def get_feature_list() -> list[str]:
    return ["all", "prefilter", "SATzilla", "SATfeatPy", "FMBA", "FM_Chara", "kbest-mutalinfo", "multisurf", "mRMR",
            "RFE",  "genetic", "HFMOEA", "embedded-tree", "SVD-entropy", "NDFS", "optuna-combined",
                #"harris-hawks",
            ]

def get_task_list() -> list[str]:
    return ["runtime_sat", "runtime_backbone", "runtime_spur", "value_ssat", "value_backbone", "algo_selection"]

def get_model_list() -> list[str]:
    return ["randomForest", "gradboostForest", "SVM", "kNN", "adaboost", "MLP"]

def parse_input() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Model Evaluation')
    parser.add_argument('pathData', type=str, help='Path to data-folder, relative to user-home')
    parser.add_argument('pathOutput', type=str, help='Path to save output to, relative to user-home')
    parser.add_argument("--features", help="Feature Subset to use, defaults to all",
                        choices=get_feature_list(),
                        default="all")
    parser.add_argument("--task", help="ML-Task to execute",
                        choices=get_task_list(), default="runtime_sat")
    parser.add_argument("--model", help="ML-Model to use",
                        choices=get_model_list(),
                        default="randomForest")
    parser.add_argument("--modelHPO", help="Optimize ML-Model using HPO (Optuna)", action=argparse.BooleanOptionalAction)
    parser.add_argument("--selectorHPO", help="Optimize Selector using HPO (Optuna)", action=argparse.BooleanOptionalAction)
    parser.add_argument("--HPOits", help="Number of HPO iterations (if modelHPO true, used for both at the same time)",
                        default=100, type=int)
    parser.add_argument("--foldNo", help="Fold to compute (starting from 0)", type=int, default=0)
    result = parser.parse_args()
    if result.features == "RFE" and not (result.model == "gradboostForest" or result.model == "randomForest" or result.model == "adaboost"):
        raise ValueError("RFE can only be used with gradboostForest, randomForest or adaboost")
    if result.features == "harris-hawks":
        raise NotImplementedError() # deactivated
    unsupported_non_HPO = ["harris-hawks", "genetic", "HFMOEA"]
    if result.features not in unsupported_non_HPO and not result.selectorHPO:
        raise ValueError(f"No HPO currently only supported for {unsupported_non_HPO}")
    if not result.selectorHPO and result.modelHPO:
        raise ValueError(f"Model HPO requires selector HPO")
    return result