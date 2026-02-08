from __future__ import annotations
from typing import Any, Dict, Optional, Tuple, List

import pandas as pd
from sklearn.preprocessing import PowerTransformer

def apply_feature_transformations(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if config is None:
        config = {"method": "none"}

    method = config.get("method", "none")

    if method == "none":
        return X_train, X_test

    if method == "yeo-johnson":
        pt = PowerTransformer(method="yeo-johnson", standardize=False)
        pt.fit(X_train.values)
        return (
            pd.DataFrame(
                pt.transform(X_train.values), 
                index=X_train.index, 
                columns=list(X_train.columns)
            ),
            pd.DataFrame(
                pt.transform(X_test.values), 
                X_test.index, 
                list(X_test.columns)
            )
        )
