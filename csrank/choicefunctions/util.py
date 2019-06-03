import copy
from itertools import product

import numpy as np
import theano
from pymc3 import Discrete
from pymc3.distributions.dist_math import bound
from theano import tensor as tt

from csrank import theano_util as ttu


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


def create_weight_dictionary(model_args, shapes):
    weights_dict = dict()
    for key, value in model_args.items():
        prior, params = copy.deepcopy(value)
        for k in params.keys():
            if isinstance(params[k], tuple):
                params[k][1]['name'] = '{}_{}'.format(key, k)
                params[k] = params[k][0](**params[k][1])
        params['name'] = key
        params['shape'] = shapes[key]
        weights_dict[key] = prior(**params)
    return weights_dict


def binary_crossentropy(p, y_true):
    if p.ndim > 1:
        l = (tt.nnet.binary_crossentropy(p, y_true).sum(axis=1)).mean()
    else:
        l = tt.nnet.binary_crossentropy(p, y_true).mean(axis=0)
    return -l


def categorical_crossentropy(p, y_true):
    return -tt.nnet.categorical_crossentropy(p, y_true)


def categorical_hinge(p, y_true):
    pos = tt.sum(y_true * p, axis=-1)
    neg = tt.max((1. - y_true) * p, axis=-1)
    return -tt.maximum(0., neg - pos + 1.)


class BinaryCrossEntropyLikelihood(Discrete):
    R"""
    Categorical log-likelihood.

    The most general discrete distribution.

    .. math:: f(x \mid p) = p_x

    ========  ===================================
    Support   :math:`x \in \{0, 1, \ldots, |p|-1\}`
    ========  ===================================

    Parameters
    ----------
    p : array of floats
        p > 0 and the elements of p must sum to 1. They will be automatically
        rescaled otherwise.
    """

    def __init__(self, p, *args, **kwargs):
        super(BinaryCrossEntropyLikelihood, self).__init__(*args, **kwargs)
        self.loss_func = categorical_hinge
        try:
            self.k = tt.shape(p)[-1].tag.test_value
        except AttributeError:
            self.k = tt.shape(p)[-1]
        self.p = tt.as_tensor_variable(p)
        self.mode = tt.argmax(p)

    def random(self, **kwargs):
        return NotImplemented

    def logp(self, value):
        p = self.p
        k = self.k
        a = self.loss_func(p, value)
        p = ttu.normalize(p)
        sum_to1 = theano.gradient.zero_grad(
            tt.le(abs(tt.sum(p, axis=-1) - 1), 1e-5))

        value_k = tt.argmax(value, axis=1)
        return bound(a, value_k >= 0, value_k <= (k - 1), sum_to1)
