import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import make_scorer, matthews_corrcoef, d2_absolute_error_score

from analysis.analysis_helper import get_pickle_dict
from helper.data_classes import TrialContainer
from helper.feature_selection import set_njobs_if_possible
from helper.model_training import is_model_classifier


def test_permutation_importance(file):
    dictonary = get_pickle_dict(file)
    trial_container: TrialContainer
    for trial_container in dictonary["trial_container"]:
        is_classifier = is_model_classifier(trial_container.model)
        scorer = make_scorer(matthews_corrcoef if is_classifier else d2_absolute_error_score, greater_is_better=True)
        model = trial_container.model
        set_njobs_if_possible(model, 1)
        bunch = permutation_importance(estimator=model, X=trial_container.x_test, y=dictonary["y_test"], scoring=scorer, n_repeats=50, n_jobs=14, random_state=42)
        mean = pd.Series(bunch["importances_mean"], index=trial_container.x_test.columns, name="mean")
        std = pd.Series(bunch["importances_std"], index=trial_container.x_test.columns, name="std")
        ret_val = {
            "importances" : pd.concat([mean, std], axis=1 ),
            "importances_raw" : pd.DataFrame(bunch["importances"], index=trial_container.x_test.columns),
        }
        yield ret_val



if __name__ == '__main__':
    path = "/home/ubuntu/MA/fe4femo/ml_analysis/copy/"
    instance = "value_ssat#genetic#gradboostForest#False#False#False#2.pkl"
    for instance in test_permutation_importance(path + instance):
        print(instance["importances"].sort_values(by="mean", axis=0, ascending=False).to_string())
