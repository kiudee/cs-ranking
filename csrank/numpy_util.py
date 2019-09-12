import numpy as np
from scipy.stats import rankdata


def replace_inf_np(x):
    if np.any(np.isinf(x)):
        x[np.isinf(x)] = 2e+300
        x[np.isnan(x)] = 2e+300
    return x


def replace_nan_np(p):
    if np.any(np.isnan(p)):
        p[np.isnan(p)] = 1.0
        p[np.isinf(p)] = 1.0
        p = normalize(p)
    return p


def logsumexp(x, axis=1):
    max_x = x.max(axis=axis, keepdims=True)
    return max_x + np.log(np.sum(np.exp(x - max_x), axis=axis, keepdims=True))


def softmax(x, axis=1):
    """
        Take softmax for the given numpy array.
        :param axis: The axis around which the softmax is applied
        :param x: array-like, shape (n_samples, ...)
        :return: softmax taken around the given axis
    """
    lse = logsumexp(x, axis=axis)
    return np.exp(x - lse)


def sigmoid(x):
    x = 1. / (1. + np.exp(-x))
    return x


def normalize(x, axis=1):
    """
        Normalize the given two dimensional numpy array around the row.
        :param axis: The axis around which the norm is applied
        :param x: theano or numpy array-like, shape (n_samples, n_objects)
        :return: normalize the array around the axis=1
    """
    return x / np.sum(x, axis=axis, keepdims=True)


def scores_to_rankings(score_matrix):
    mask3 = np.equal(score_matrix[:, None] - score_matrix[:, :, None], 0)
    n_objects = score_matrix.shape[1]
    ties = np.sum(np.sum(mask3, axis=(1, 2)) - n_objects)
    rankings = np.empty_like(score_matrix)
    if ties > 0:
        for i, s in enumerate(score_matrix):
            rankings[i] = len(s) - rankdata(s)
    else:
        orderings = np.argsort(score_matrix, axis=1)[:, ::-1]
        rankings = np.argsort(orderings, axis=1)
    return rankings


def ranking_ordering_conversion(input):
    output = np.argsort(input, axis=1)
    return output
