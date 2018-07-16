import logging
from itertools import combinations

import numpy as np

from csrank.numpy_util import ranking_ordering_conversion

__all__ = ['generate_complete_pairwise_dataset', 'complete_linear_regression_dataset',
           'complete_linear_regression_dataset', "sub_sampling_rankings"]


def generate_pairwise_instances(features):
    pairs = np.array(list(combinations(features, 2)))

    n_pairs = len(pairs)
    neg_indices = np.arange(0, n_pairs, 2)

    a, b = np.copy(pairs[neg_indices, 0]), np.copy(pairs[neg_indices, 1])
    pairs[neg_indices, 1] = a
    pairs[neg_indices, 0] = b

    X1 = pairs[:, 0]
    X2 = pairs[:, 1]
    Y_double = np.ones([n_pairs, 1]) * np.array([1, 0])
    Y_single = np.repeat(1, n_pairs)

    Y_double[neg_indices] = [0, 1]
    Y_single[neg_indices] = 0
    return X1, X2, Y_double, Y_single


def generate_complete_pairwise_dataset(X, rankings):
    try:
        n_instances, n_objects, n_features = X.shape
        rankings = rankings.astype(int)
        rankings -= np.min(rankings)
        orderings = ranking_ordering_conversion(rankings)
        del rankings
        x_sorted = [X[i, orderings[i], :] for i in range(n_instances)]
        del orderings
    except ValueError:
        # TODO Add the code to change the rankings to orderings and sort X according to that
        logger = logging.getLogger("generate_complete_pairwise_dataset")
        logger.error("Value Error: {}, {} ".format(X[0], rankings[0]))
        x_sorted = X
    y_double = []
    x_train1 = []
    x_train2 = []
    y_single = []
    for features in x_sorted:
        x1, x2, y1, y2 = generate_pairwise_instances(features)
        x_train1.extend(x1)
        x_train2.extend(x2)
        y_double.extend(y1)
        y_single.extend(y2)
    x_train1 = np.array(x_train1)
    x_train2 = np.array(x_train2)
    y_double = np.array(y_double)
    y_single = np.array(y_single)
    x_train = x_train1 - x_train2
    return x_train, x_train1, x_train2, y_double, y_single


def complete_linear_regression_dataset(X, rankings):
    X1 = []
    Y_single = []
    for features, rank in zip(X, rankings):
        X1.extend(features)
        norm_ranks = rank / np.max(rank, axis=0)
        Y_single.extend(norm_ranks)
    X1 = np.array(X1)
    Y_single = np.array(Y_single)
    return X1, Y_single


def sub_sampling_rankings(Xt, Yt, n_objects=5):
    bucket_size = int(Xt.shape[1] / n_objects)
    X_train = []
    Y_train = []
    for i in range(bucket_size):
        X = np.copy(Xt)
        Y = np.copy(Yt)
        rs = np.random.RandomState(42 + i)
        idx = rs.randint(bucket_size, size=(len(X), n_objects))
        idx += np.arange(start=0, stop=X.shape[1], step=bucket_size)[:n_objects]
        X = X[np.arange(X.shape[0])[:, None], idx]
        Y = Y[np.arange(X.shape[0])[:, None], idx]
        tmp_sort = Y.argsort(axis=-1)
        Y = np.empty_like(Y)
        Y[np.arange(len(X))[:, None], tmp_sort] = np.arange(n_objects)
        if len(X_train) == 0:
            X_train = X
            Y_train = Y
        else:
            Y_train = np.concatenate([Y_train, Y], axis=0)
            X_train = np.concatenate([X_train, X], axis=0)
    return X_train, Y_train
