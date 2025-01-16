import pandas as pd
import numpy as np
from scipy.linalg import svdvals


def compute_dataset_entropy(matrix : np.ndarray) -> float:
    svdval = svdvals(matrix)
    rank = np.linalg.matrix_rank(matrix)
    eigenvalues = svdval * svdval
    cumulator = 0
    eigenvalues_sum = sum(eigenvalues)
    for i in range(rank):
        eigenvalue = eigenvalues[i]
        if eigenvalue == 0:
            continue
        vj = eigenvalue / eigenvalues_sum
        cumulator += vj * np.log2(vj)
    return -1 * (1/ np.log2(rank)) * cumulator

def compute_feature_contribution(matrix : np.ndarray, position : int, full_entropy : float = None) -> float:
    if full_entropy is None:
        full_entropy = compute_dataset_entropy(matrix)
    partial_entropy = compute_dataset_entropy(np.delete(matrix, position, 1))
    return full_entropy - partial_entropy

def keep_high_contrib_features(df : pd.DataFrame) -> list[bool]:
    np_view = df.to_numpy()
    full_entropy = compute_dataset_entropy(np_view)
    contributions = np.array([compute_feature_contribution(np_view, i, full_entropy) for i in range(np_view.shape[1])])
    min_acceptance = contributions.mean() + contributions.std()
    bool_mask = [ x > min_acceptance for x in contributions]
    return bool_mask

