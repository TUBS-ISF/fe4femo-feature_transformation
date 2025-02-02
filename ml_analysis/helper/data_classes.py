from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, eq=True)
class FoldSplit:
    fold_no: int
    train_index: np.ndarray
    test_index: np.ndarray