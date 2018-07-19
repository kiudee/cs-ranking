from itertools import product

import numpy as np


def generate_pairwise_instances(x, choice):
    pairs = np.array(list(product(choice, x)) + list(product(x, choice)))
    n_pairs = len(pairs)
    neg_indices = np.arange(int(n_pairs / 2), n_pairs)
    X1 = pairs[:, 0]
    X2 = pairs[:, 1]
    Y_double = np.ones([n_pairs, 1]) * np.array([1, 0])
    Y_single = np.repeat(1, n_pairs)

    Y_double[neg_indices] = [0, 1]
    Y_single[neg_indices] = 0
    return X1, X2, Y_double, Y_single


def generate_complete_pairwise_dataset(X, Y):
    Y_double = []
    X1 = []
    X2 = []
    Y_single = []
    # Y = np.where(Y==1)
    for x, y in zip(X, Y):
        choice = x[y == 1]
        x = np.delete(x, np.where(y == 1)[0], 0)
        if len(choice) != 0 and len(x) != 0:
            x1, x2, y1, y2 = generate_pairwise_instances(x, choice)
            X1.extend(x1)
            X2.extend(x2)
            Y_double.extend(y1)
            Y_single.extend(y2)
    X1 = np.array(X1)
    X2 = np.array(X2)
    Y_double = np.array(Y_double)
    Y_single = np.array(Y_single)
    X_train = X1 - X2
    return X1, X2, X_train, Y_double, Y_single
