from sklearn.base import is_classifier
from sklearn.metrics import matthews_corrcoef, d2_absolute_error_score

from analysis.helper import get_pickle_dict

path = "/home/ubuntu/MA/fe4femo/ml_analysis/copy/"
file = "runtime_backbone#HFMOEA#randomForest#True#True#50#0.pkl"
dictonary = get_pickle_dict(path, file)

y_pred = dictonary["model"].predict(dictonary["X_test"])
y_test = dictonary["y_test"]
if is_classifier(dictonary["model"]):
    print(matthews_corrcoef(y_test, y_pred))
else:
    print(d2_absolute_error_score(y_test, y_pred))