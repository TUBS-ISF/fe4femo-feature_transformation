from dataclasses import dataclass

import numpy as np
from sklearn.base import BaseEstimator


@dataclass(frozen=True, eq=True)
class FoldSplit:
    fold_no: int
    train_index: np.ndarray
    test_index: np.ndarray

@dataclass(frozen=True, eq=True)
class FoldResult:
    model_quality: float
    feature_computation_time: float

@dataclass(frozen=True)
class TrialContainer:
    model: BaseEstimator
    best_params: dict[str, any]
    time_Feature: float
    time_Model: float