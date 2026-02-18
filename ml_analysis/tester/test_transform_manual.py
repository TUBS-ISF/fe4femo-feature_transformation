import pandas as pd

from sklearn.metrics import r2_score, matthews_corrcoef
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import LabelEncoder

from ml_analysis.helper.load_dataset import get_dataset, generate_xy_split, is_task_classification
from ml_analysis.helper.feature_selection import impute_and_scale  # <- now impute-only after your change
from ml_analysis.helper.feature_transformation import apply_feature_transformations


# ----------------------------
# Settings
# ----------------------------
pathData = "/mnt/e/Uni/Thesis/fe4femo-feature_transformation/data"
task = "runtime_backbone"
foldNo = 0

# Baseline + exactly what you asked for:
TRANSFORMS = [
    #{"method": "none"},              # baseline
    #{"method": "standardize"},       # scaling: standardization
    #{"method": "minmax"},            # scaling: normalization
    #{"method": "log1p"},             # nonlinear: log (safe log in helper)
    #{"method": "quantile-normal", "n_quantiles": 1000, "random_state": 0},  # quantile
    #{"method": "yeo-johnson"},       # power
    {"method": "pca", "pca_var": 0.95, "random_state": 0},                 # dim reduction (needs standardize)
    {"method": "nystroem-rbf", "n_components": 200, "gamma": None, "random_state": 0},  # kernel (needs standardize)
    {"method": "bin-onehot", "n_bins": 10, "strategy": "quantile"},         # encoding
]

# Which transforms require standardization FIRST (distance/variance based)
REQUIRES_STANDARDIZE = {"pca", "nystroem-rbf"}


# ----------------------------
# Load + split
# ----------------------------
X, y = get_dataset(pathData, task)
is_cls = is_task_classification(task)

if is_cls:
    le = LabelEncoder()
    y = pd.Series(le.fit_transform(y), index=y.index)

X_train, X_test, y_train, y_test = generate_xy_split(X, y, f"{pathData}/folds.txt", foldNo)

# Mandatory preprocessing: impute only (your edited function)
X_train, X_test = impute_and_scale(X_train, X_test)


# ----------------------------
# Models (use at least 1 linear model, otherwise PCA/kernel won’t be meaningful)
# ----------------------------
if is_cls:
    MODELS = [
        ("LogReg", LogisticRegression(max_iter=3000)),
        ("RF", RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1)),
    ]
else:
    MODELS = [
        ("Ridge", Ridge(alpha=1.0, random_state=0)),
        ("RF", RandomForestRegressor(n_estimators=300, random_state=0, n_jobs=-1)),
    ]


def score_model(model_name, model, Xt_train, Xt_test, y_train, y_test, is_cls):
    model.fit(Xt_train, y_train)
    pred = model.predict(Xt_test)
    if is_cls:
        return "MCC", matthews_corrcoef(y_test, pred)
    else:
        return "R2", r2_score(y_test, pred)


# ----------------------------
# Run manual tests
# ----------------------------
for cfg in TRANSFORMS:
    method = cfg["method"]

    # Enforce required standardization *before* PCA/kernel
    if method in REQUIRES_STANDARDIZE:
        X0_train, X0_test = apply_feature_transformations(X_train, X_test, {"method": "standardize"})
        Xt_train, Xt_test = apply_feature_transformations(X0_train, X0_test, cfg)
        pipeline_name = f"standardize -> {method}"
    else:
        Xt_train, Xt_test = apply_feature_transformations(X_train, X_test, cfg)
        pipeline_name = method

    # Print shapes so you see reduction/expansion
    shape_str = f"{Xt_train.shape[0]}x{Xt_train.shape[1]}"

    for model_name, model in MODELS:
        metric_name, val = score_model(model_name, model, Xt_train, Xt_test, y_train, y_test, is_cls)
        print(f"{pipeline_name:>22} | {model_name:<6} | {metric_name}={val:.4f} | X={shape_str}")
