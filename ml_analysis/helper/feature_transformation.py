from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import pandas as pd

from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    QuantileTransformer,
    PowerTransformer,
    KBinsDiscretizer,
    OneHotEncoder,
)
from sklearn.decomposition import PCA
from sklearn.kernel_approximation import Nystroem


def _df(arr: np.ndarray, index: pd.Index, columns: List[str]) -> pd.DataFrame:
    return pd.DataFrame(arr, index=index, columns=columns)


def _safe_log1p(X: pd.DataFrame) -> pd.DataFrame:
    """
    log1p requires all values > -1.
    If min <= -1, we shift the whole matrix upward so min becomes (-1 + eps),
    then apply log1p. This avoids crashes and keeps a monotonic mapping.
    """
    
    Xv = X.to_numpy(dtype=np.float64, copy=True)
    print("inf count:", np.isinf(Xv).sum(), "max:", np.max(Xv))
    Xv[~np.isfinite(Xv)] = 0.0

    m = Xv.min()
    if m <= -1:
        shift = (-1 - m) + 1e-6
        Xv = Xv + shift
    return _df(np.log1p(Xv), X.index, list(X.columns))

def _safe_log1p_pair(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    print("train min:", X_train.min().min(), "test min:", X_test.min().min())

    Xtr = X_train.to_numpy(dtype=np.float64, copy=True)
    Xte = X_test.to_numpy(dtype=np.float64, copy=True)

    # compute shift from train
    m = Xtr.min()
    shift = 0.0
    if m <= -1.0:
        shift = (-1.0 - m) + 1e-6

    Xtr = np.log1p(Xtr + shift)
    Xte = np.log1p(Xte + shift)

    return (
        _df(Xtr, X_train.index, list(X_train.columns)),
        _df(Xte, X_test.index, list(X_test.columns)),
    )

def apply_feature_transformations(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if config is None:
        config = {"method": "none"}

    method = config.get("method", "none")

    # ---------------- Baseline ----------------
    if method == "none":
        return X_train, X_test

    # ---------------- Scaling ----------------
    if method == "standardize":
        scaler = StandardScaler()
        scaler.fit(X_train.values)
        return (
            _df(scaler.transform(X_train.values), X_train.index, list(X_train.columns)),
            _df(scaler.transform(X_test.values), X_test.index, list(X_test.columns)),
        )

    if method == "minmax":
        scaler = MinMaxScaler()
        scaler.fit(X_train.values)
        return (
            _df(scaler.transform(X_train.values), X_train.index, list(X_train.columns)),
            _df(scaler.transform(X_test.values), X_test.index, list(X_test.columns)),
        )

    # ---------------- Nonlinear (log) ----------------
    if method == "log1p":
        return _safe_log1p_pair(X_train, X_test)
    
    if method == "signed-log1p":
        Xtr = X_train.to_numpy(dtype=np.float64, copy=False)
        Xte = X_test.to_numpy(dtype=np.float64, copy=False)

        Ztr = np.sign(Xtr) * np.log1p(np.abs(Xtr))
        Zte = np.sign(Xte) * np.log1p(np.abs(Xte))

        return (
            _df(Ztr, X_train.index, list(X_train.columns)),
            _df(Zte, X_test.index, list(X_test.columns)),
        )

    # ---------------- Quantile ----------------
    if method == "quantile-normal":
        n_quantiles = int(config.get("n_quantiles", min(200, len(X_train))))
        qt = QuantileTransformer(
            n_quantiles=n_quantiles,
            output_distribution="normal",
            random_state=int(config.get("random_state", 0)),
        )
        qt.fit(X_train.values)
        return (
            _df(qt.transform(X_train.values), X_train.index, list(X_train.columns)),
            _df(qt.transform(X_test.values), X_test.index, list(X_test.columns)),
        )

    # ---------------- Power (Yeo–Johnson) ----------------
    if method == "yeo-johnson":

       # work in float64 for numerical stability
        Xtr = X_train.to_numpy(dtype=np.float64, copy=False)
        Xte = X_test.to_numpy(dtype=np.float64, copy=False)

        # replace inf with nan, then impute per column using train median
        Xtr = np.where(np.isfinite(Xtr), Xtr, np.nan)
        Xte = np.where(np.isfinite(Xte), Xte, np.nan)
        med = np.nanmedian(Xtr, axis=0)
        Xtr = np.where(np.isnan(Xtr), med, Xtr)
        Xte = np.where(np.isnan(Xte), med, Xte)

        Xtr_out = Xtr.copy()
        Xte_out = Xte.copy()

        for j in range(Xtr.shape[1]):
            col_tr = Xtr[:, j]
            col_te = Xte[:, j]

            # skip constant / near-constant
            if np.std(col_tr) < 1e-12:
                continue

            try:
                pt = PowerTransformer(method="yeo-johnson", standardize=False)
                pt.fit(col_tr.reshape(-1, 1))

                ztr = pt.transform(col_tr.reshape(-1, 1)).ravel()
                zte = pt.transform(col_te.reshape(-1, 1)).ravel()

                # accept only finite results
                if np.isfinite(ztr).all() and np.isfinite(zte).all():
                    Xtr_out[:, j] = ztr
                    Xte_out[:, j] = zte
            except Exception:
                # leave unchanged on failure
                continue

        # enforce finite
        Xtr_out = np.nan_to_num(Xtr_out, nan=0.0, posinf=0.0, neginf=0.0)
        Xte_out = np.nan_to_num(Xte_out, nan=0.0, posinf=0.0, neginf=0.0)

        # IMPORTANT: prevent later float32 overflow -> inf
        f32_max = np.finfo(np.float32).max / 10  # margin
        Xtr_out = np.clip(Xtr_out, -f32_max, f32_max)
        Xte_out = np.clip(Xte_out, -f32_max, f32_max)

        return (
            _df(Xtr_out, X_train.index, list(X_train.columns)),
            _df(Xte_out, X_test.index, list(X_test.columns)),
        )


    # ---------------- Dimensionality reduction ----------------
    if method == "pca":
        var = float(config.get("pca_var", 0.95))
        pca = PCA(n_components=var, random_state=int(config.get("random_state", 0)))
        pca.fit(X_train.values)
        Xtr = pca.transform(X_train.values)
        Xte = pca.transform(X_test.values)
        cols = [f"pca_{i}" for i in range(Xtr.shape[1])]
        return _df(Xtr, X_train.index, cols), _df(Xte, X_test.index, cols)

    # ---------------- Kernel approximation (Nystroem RBF) ----------------
    if method == "nystroem-rbf":
        X0_train, X0_test = apply_feature_transformations(X_train, X_test, {"method": "standardize"})
        n_components = int(config.get("n_components", 300))
        gamma = config.get("gamma", None)  # None lets sklearn pick (1/n_features)
        nys = Nystroem(
            kernel="rbf",
            n_components=n_components,
            gamma=gamma,
            random_state=int(config.get("random_state", 0)),
        )
        nys.fit(X0_train.values)
        Xtr = nys.transform(X0_train.values)
        Xte = nys.transform(X0_test.values)
        cols = [f"nystroem_{i}" for i in range(Xtr.shape[1])]
        return _df(Xtr, X_train.index, cols), _df(Xte, X_test.index, cols)

    # ---------------- Encoding (bin) ----------------
    if method == "bin-ordinal":        
        n_bins = int(config.get("n_bins", 5))              # start smaller than 10
        strategy = config.get("strategy", "uniform")       # start with uniform (fastest)

        binner = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy=strategy)
        binner.fit(X_train.values)

        Xtr = binner.transform(X_train.values)
        Xte = binner.transform(X_test.values)

        cols = [f"bin_{c}" for c in range(Xtr.shape[1])]
        return _df(Xtr, X_train.index, cols), _df(Xte, X_test.index, cols)


    raise ValueError(f"Unknown transform method: {method}")
