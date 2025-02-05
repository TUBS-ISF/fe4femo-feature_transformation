from sklearn.base import is_classifier
from sklearn.metrics import matthews_corrcoef, d2_absolute_error_score

from analysis.helper import get_pickle_dict

def get_model_quality(file):
    dictonary = get_pickle_dict(file)

    y_pred = dictonary["model"].predict(dictonary["X_test"])
    y_test = dictonary["y_test"]
    if is_classifier(dictonary["model"]):
        return matthews_corrcoef(y_test, y_pred)
    else:
        return d2_absolute_error_score(y_test, y_pred)

if __name__ == '__main__':
    path = "/home/ubuntu/MA/fe4femo/ml_analysis/copy/"
    instance = "runtime_backbone#HFMOEA#randomForest#True#True#50#0.pkl"
    print(get_model_quality(path + instance))