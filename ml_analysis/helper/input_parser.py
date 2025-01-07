import argparse


def parse_input() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Model Evaluation')
    parser.add_argument('pathData', type=str, help='Path to data-folder')
    parser.add_argument('pathOutput', type=str, help='Path to save output to')
    parser.add_argument("--features", help="Feature Subset to use, defaults to all",
                        choices=["all", "SATzilla", "SATfeatPy", "FMBA", "FM_Chara",
                                 "kbest-mutalinfo", "multisurf", "mRMR", "SFS-forward", "SFS-backward", "harris-hawks",
                                 "genetic", "HFMOEA", "embedded-tree", "SVD-entropy", "NDFS", "optuna-combined"],
                        default="all")
    parser.add_argument("--task", help="ML-Task to execute",
                        choices=["runtime_sat", "runtime_backbone", "runtime_spur", "value_ssat", "value_backbone",
                                 "algo_selection"], default="runtime_sat")
    parser.add_argument("--model", help="ML-Model to use",
                        choices=["randomForest", "gradboostForest", "SVM", "kNN", "adaboost", "MLP"],
                        default="randomForest")
    parser.add_argument("--modelHPO", help="Optimize ML-Model using HPO (Optuna)", default=False, action="store_true")
    parser.add_argument("--HPOits", help="Number of HPO iterations (if modelHPO true, used for both at the same time)",
                        default=100, type=int)
    parser.add_argument("--foldNo", help="Fold to compute (starting from 0)", type=int, default=0)

    return parser.parse_args()