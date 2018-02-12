import logging
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from csrank.util import ranking_ordering_conversion

__all__ = ['generate_complete_pairwise_dataset', 'complete_linear_regression_dataset',
           'complete_linear_regression_dataset', "weighted_cosine_similarity", "get_key_for_indices"]


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
        X_sorted = [X[i, orderings[i], :] for i in range(n_instances)]
    except ValueError:
        # TODO Add the code to change the rankings to orderings and sort X according to that
        logger = logging.getLogger("generate_complete_pairwise_dataset")
        logger.error("Value Error: {}, {} ".format(X[0], rankings[0]))
        X_sorted = X
    Y_double = []
    X1 = []
    X2 = []
    Y_single = []
    for features in X_sorted:
        x1, x2, y1, y2 = generate_pairwise_instances(features)
        X1.extend(x1)
        X2.extend(x2)
        Y_double.extend(y1)
        Y_single.extend(y2)
    X1 = np.array(X1)
    X2 = np.array(X2)
    Y_double = np.array(Y_double)
    Y_single = np.array(Y_single)
    X_train = X1 - X2
    return X_train, X1, X2, Y_double, Y_single


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


def get_key_for_indices(idx1, idx2):
    return str(tuple(sorted([idx1, idx2])))


def weighted_cosine_similarity(weights, x, y):
    denominator = np.sqrt(np.sum(weights * x * x)) * np.sqrt(
        np.sum(weights * y * y))
    sim = np.sum(weights * x * y) / denominator
    return sim


def similarity_function_for_multilabel_instances(X_labels, Y_labels, X, Y):
    similarity = f1_score(X_labels, Y_labels, average='macro')
    similarity = np.dot(X, Y) / (np.linalg.norm(X) * np.linalg.norm(Y)) + similarity
    return similarity


def initialize_similarity_matrix(mypath):
    dataFrame = pd.read_csv(mypath)
    similarity_dictionary = dataFrame.set_index('col_major_index')['similarity'].to_dict()
    return similarity_dictionary


def sub_sampling(name, Xt, Yt, n_objects=5):
    logger = logging.getLogger(name=name)
    bucket_size = int(Xt.shape[1] / n_objects)
    #logger.info("#########################################################################")
    #logger.info("X instances {} objects {} bucket_size {}".format(Xt.shape[0], Xt.shape[1], bucket_size))
    X_train = []
    Y_train = []
    for i in range(bucket_size):
        X = np.copy(Xt)
        Y = np.copy(Yt)
        rs = np.random.RandomState(42 + i)
        idx = rs.randint(bucket_size, size=(len(X), n_objects))
        # TODO: subsampling multiple rankings
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
    logger.info("Sampled instances {} objects {}".format(X_train.shape[0], X_train.shape[1]))
    return X_train, Y_train
