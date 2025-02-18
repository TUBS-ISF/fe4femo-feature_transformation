from sklearn.base import is_classifier
from sklearn.metrics import matthews_corrcoef, d2_absolute_error_score

from analysis.analysis_helper import get_pickle_dict
from helper.model_training import is_model_classifier


def get_model_quality(file):
    dictonary = get_pickle_dict(file)
    ret_list = []
    for trial_container in dictonary["trial_container"]:
        y_pred = trial_container.model.predict(trial_container.x_test)
        y_test = dictonary["y_test"]
        if is_model_classifier(trial_container.model):
            ret_list.append(matthews_corrcoef(y_test, y_pred).item())
        else:
            ret_list.append(d2_absolute_error_score(y_test, y_pred).item())
    return ret_list

if __name__ == '__main__':
    path = "/home/ubuntu/MA/fe4femo/ml_analysis/copy/"
    instance = "runtime_backbone#HFMOEA#randomForest#True#True#50#0.pkl"
    print(get_model_quality(path + instance))