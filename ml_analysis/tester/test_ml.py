import pandas as pd
from optuna.trial import FrozenTrial, FixedTrial
from sklearn.metrics import matthews_corrcoef, d2_absolute_error_score
from sklearn.preprocessing import LabelEncoder

from generate_fold_model import impute_and_scale
from helper.feature_selection import get_selection_HPO_space, get_feature_selection
from helper.load_dataset import load_algo_selection, load_feature_data, generate_xy_split, load_dataset, \
    is_task_classification, load_feature_groups, get_dataset
from helper.model_training import get_model_HPO_space, get_model




task = "runtime_backbone"
model = "randomForest"
features = "HFMOEA"
modelHPO = True
feature_groups = load_feature_groups("/home/ubuntu/MA/data")
n_jobs = 12
frozen_best_trial = FixedTrial({
    "n_estimators" : 100,
    "max_depth" : 100,

    "topk" : 10,
    "pop_size" : 40,
    "mutation_probability" : 0.1,
})

X,y = get_dataset("/home/ubuntu/MA/data", task)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
y = pd.Series(y)

X_train, X_test, y_train, y_test = generate_xy_split(X, y, "/home/ubuntu/MA/data/folds.txt", 0)

X_train, X_test = impute_and_scale(X_train, X_test)

is_classification = is_task_classification(task)
model_config = get_model_HPO_space(model, frozen_best_trial, is_classification) if modelHPO else None
selector_config = get_selection_HPO_space(features, frozen_best_trial, is_classification, feature_groups, X_train.shape[1])
model_instance_selector = get_model(model, is_classification, 1, model_config)
X_train, X_test = get_feature_selection(features, is_classification, X_train, y_train, X_test,
                                        selector_config, model_instance_selector, feature_groups, parallelism=n_jobs)
print(X_train)
print(X_test)
model_instance = get_model(model, is_classification, n_jobs, model_config)
model_instance.fit(X_train, y_train)
y_pred = model_instance.predict(X_test)
if is_classification:
    print(matthews_corrcoef(y_test, y_pred))
else:
    print(d2_absolute_error_score(y_test, y_pred))