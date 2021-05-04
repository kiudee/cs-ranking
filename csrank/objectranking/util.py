import numpy as np


__all__ = ["complete_linear_regression_dataset"]


def complete_linear_regression_dataset(X, rankings):
    X1 = []
    Y_single = []
    for features, rank in zip(X, rankings):
        X1.extend(features)
        min_rank = np.min(rank, axis=0)
        max_rank = np.max(rank, axis=0)
        norm_ranks = (rank - min_rank + 1) / (max_rank - min_rank + 1)
        Y_single.extend(norm_ranks)
    X1 = np.array(X1)
    Y_single = np.array(Y_single)
    return X1, Y_single
