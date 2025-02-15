from optuna_dashboard import run_server

from analysis.analysis_helper import get_optuna_study

if __name__ == '__main__':
    path = "/home/ubuntu/MA/fe4femo/ml_analysis/copy/"
    instance = "value_ssat#optuna-combined#randomForest#True#True#150#True#0.journal"
    study_name = instance.replace(".journal", "")
    study, journal = get_optuna_study(path+instance, study_name=study_name)
    run_server(storage=journal)
