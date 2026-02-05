# ml_analysis/tester/test_transformations_suite.py
# Minimal, portable playground for "1 method per category" feature transformations
# + compares a few models. No project data needed.

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    QuantileTransformer,
    PowerTransformer,
    KBinsDiscretizer,
)
from sklearn.decomposition import PCA
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_absolute_error


# -------------------------
# 1) Synthetic FM-ish dataset
# -------------------------
def make_synthetic_df(n=600, d=40, seed=42):
    """
    Makes mixed-scale, skewed, partly heavy-tailed numeric data,
    similar in spirit to "FM-derived metrics" (counts/ratios/depths/etc.).
    """
    rng = np.random.default_rng(seed)
    X = np.empty((n, d), dtype=float)

    for j in range(d):
        if j % 4 == 0:
            # count-like heavy-tailed positive
            X[:, j] = rng.lognormal(mean=0.0, sigma=1.2, size=n)
        elif j % 4 == 1:
            # ratio-like bounded-ish
            X[:, j] = rng.beta(a=2.0, b=5.0, size=n) * 10.0
        elif j % 4 == 2:
            # normal-ish
            X[:, j] = rng.normal(loc=0.0, scale=2.0, size=n)
        else:
            # wide uniform with negatives
            X[:, j] = rng.uniform(low=-3.0, high=3.0, size=n)

    cols = [f"f{j}" for j in range(d)]
    X = pd.DataFrame(X, columns=cols)

    # Inject some missingness (FM datasets often have missing/failed measurements)
    mask = rng.random(X.shape) < 0.02
    X = X.mask(mask)

    # Nonlinear target with log-ish + interactions (proxy for runtime-like behavior)
    y = (
        0.7 * np.log1p(np.abs(X["f0"].fillna(0))) +
        0.15 * (X["f1"].fillna(0) ** 2) +
        0.10 * (X["f2"].fillna(0) * X["f3"].fillna(0)) -
        0.08 * X["f4"].fillna(0) +
        rng.normal(0, 0.25, size=n)
    )
    y = pd.Series(y, name="y")
    return X, y


# -------------------------
# 2) Core preprocessing
# -------------------------
def impute(X_train: pd.DataFrame, X_test: pd.DataFrame):
    imp = SimpleImputer(strategy="median")
    Xtr = imp.fit_transform(X_train)
    Xte = imp.transform(X_test)
    return (
        pd.DataFrame(Xtr, index=X_train.index, columns=X_train.columns),
        pd.DataFrame(Xte, index=X_test.index, columns=X_test.columns),
    )


# -------------------------
# 3) Transformation suite
#    (1 representative per category, plus baseline)
# -------------------------
class TransformSuite:
    """
    Each 'method' here is a full operator you can compare.
    All fitting is done on X_train ONLY.
    Returns DataFrames and preserves index.
    """

    def __init__(self, method: str, random_state: int = 0):
        self.method = method
        self.random_state = random_state
        self._sk = None
        self._shift = 0.0

    def fit(self, X_train: pd.DataFrame):
        m = self.method

        # Baseline: do nothing (after imputation)
        if m == "none":
            return self

        # Baselines / Scaling category
        if m == "standardize":
            self._sk = StandardScaler()
            self._sk.fit(X_train.values)
            return self

        if m == "minmax":
            self._sk = MinMaxScaler()
            self._sk.fit(X_train.values)
            return self

        # Function transformation category (representative: log1p)
        if m == "log1p+standardize":
            # log1p is stateless, standardize is fitted
            self._sk = StandardScaler()
            Xtr = self._log1p_safe(X_train).values
            self._sk.fit(Xtr)
            return self

        # Power transformation category (representative: Yeo–Johnson)
        if m == "yeo-johnson+standardize":
            self._pt = PowerTransformer(method="yeo-johnson", standardize=False)
            self._pt.fit(X_train.values)
            Xtr = self._pt.transform(X_train.values)
            self._sk = StandardScaler()
            self._sk.fit(Xtr)
            return self

        # Quantile transformation category
        if m == "quantile(normal)+standardize":
            self._qt = QuantileTransformer(
                output_distribution="normal",
                n_quantiles=min(200, len(X_train)),
                random_state=self.random_state,
            )
            self._qt.fit(X_train.values)
            Xtr = self._qt.transform(X_train.values)
            self._sk = StandardScaler()
            self._sk.fit(Xtr)
            return self

        # Dimensionality reduction category (representative: PCA)
        if m == "pca(0.95)+standardize":
            self._sc = StandardScaler()
            Xtr = self._sc.fit_transform(X_train.values)
            self._pca = PCA(n_components=0.95, random_state=self.random_state)
            self._pca.fit(Xtr)
            return self

        # Discretization category (representative: KBins)
        if m == "kbins(10,quantile)+standardize":
            self._sc = StandardScaler()
            Xtr = self._sc.fit_transform(X_train.values)
            self._kb = KBinsDiscretizer(
                n_bins=10, encode="onehot-dense", strategy="quantile"
            )
            self._kb.fit(Xtr)
            return self

        # Kernel approximation category (representative: RBF sampler)
        if m == "rbf(300)+standardize":
            self._sc = StandardScaler()
            Xtr = self._sc.fit_transform(X_train.values)
            self._rbf = RBFSampler(
                gamma=1.0, n_components=300, random_state=self.random_state
            )
            self._rbf.fit(Xtr)
            return self

        raise ValueError(f"Unknown method: {m}")

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        m = self.method

        if m == "none":
            return X

        if m in ("standardize", "minmax"):
            Xt = self._sk.transform(X.values)
            return pd.DataFrame(Xt, index=X.index, columns=X.columns)

        if m == "log1p+standardize":
            Xlog = self._log1p_safe(X).values
            Xt = self._sk.transform(Xlog)
            return pd.DataFrame(Xt, index=X.index, columns=X.columns)

        if m == "yeo-johnson+standardize":
            Xyj = self._pt.transform(X.values)
            Xt = self._sk.transform(Xyj)
            return pd.DataFrame(Xt, index=X.index, columns=X.columns)

        if m == "quantile(normal)+standardize":
            Xq = self._qt.transform(X.values)
            Xt = self._sk.transform(Xq)
            return pd.DataFrame(Xt, index=X.index, columns=X.columns)

        if m == "pca(0.95)+standardize":
            Xsc = self._sc.transform(X.values)
            Xp = self._pca.transform(Xsc)
            cols = [f"pca_{i}" for i in range(Xp.shape[1])]
            return pd.DataFrame(Xp, index=X.index, columns=cols)

        if m == "kbins(10,quantile)+standardize":
            Xsc = self._sc.transform(X.values)
            Xb = self._kb.transform(Xsc)
            cols = [f"bin_{i}" for i in range(Xb.shape[1])]
            return pd.DataFrame(Xb, index=X.index, columns=cols)

        if m == "rbf(300)+standardize":
            Xsc = self._sc.transform(X.values)
            Xk = self._rbf.transform(Xsc)
            cols = [f"rbf_{i}" for i in range(Xk.shape[1])]
            return pd.DataFrame(Xk, index=X.index, columns=cols)

        raise ValueError(f"Unknown method: {m}")

    @staticmethod
    def _log1p_safe(X: pd.DataFrame) -> pd.DataFrame:
        # log1p requires values > -1; shift if needed (keeps monotonicity).
        Xv = X.copy()
        min_val = np.nanmin(Xv.values)
        if min_val <= -1.0:
            shift = (-1.0 - min_val) + 1e-6
            Xv = Xv + shift
        return np.log1p(Xv)


def apply_method(X_train, X_test, method: str, random_state: int = 0):
    t = TransformSuite(method=method, random_state=random_state).fit(X_train)
    return t.transform(X_train), t.transform(X_test)


# -------------------------
# 4) Models to compare
# -------------------------
def get_models():
    return {
        "Ridge": Ridge(alpha=1.0, random_state=0),
        # "SVR(RBF)": SVR(C=10.0, gamma="scale"),
        #"RandomForest": RandomForestRegressor(
        #    n_estimators=300, random_state=0, n_jobs=-1
        #),
    }


# -------------------------
# 5) Run the benchmark
# -------------------------
if __name__ == "__main__":
    X, y = make_synthetic_df()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    # Always do imputation first (fit on train only)
    X_train, X_test = impute(X_train, X_test)

    # Your chosen "1 per category" (+ baseline scaling variants)
    methods = [
        "none",
        "standardize",                 # Scaling (representative)
        "minmax",                      # Scaling alternative (optional)
        "log1p+standardize",           # Function transform (representative)
        "yeo-johnson+standardize",     # Power transform (representative)
        "quantile(normal)+standardize",# Quantile transform (representative)
        "pca(0.95)+standardize",       # Dimensionality reduction (representative)
        "kbins(10,quantile)+standardize",# Discretization (representative)
        "rbf(300)+standardize",        # Kernel approximation (representative)
    ]

    models = get_models()

    print("\n=== Minimal Transformation Benchmark (synthetic FM-ish data) ===\n")
    for method in methods:
        Xt_train, Xt_test = apply_method(X_train, X_test, method=method, random_state=0)

        # sanity checks
        assert (Xt_train.index == X_train.index).all()
        assert (Xt_test.index == X_test.index).all()

        print(f"\n--- method: {method} ---")
        print(f"Shapes: train={Xt_train.shape}, test={Xt_test.shape}")

        for name, model in models.items():
            model.fit(Xt_train, y_train)
            pred = model.predict(Xt_test)
            r2 = r2_score(y_test, pred)
            mae = mean_absolute_error(y_test, pred)
            print(f"{name:12s}  R2={r2:.4f}  MAE={mae:.4f}")
