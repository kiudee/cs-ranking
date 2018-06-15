import numpy as np
from theano import tensor as tt

from csrank.util import normalize


def replace_inf_theano(x):
    if tt.any(tt.isinf(x)):
        x = tt.switch(tt.isinf(x), 2e+300, x)
        x = tt.switch(tt.isnan(x), 2e+300, x)
    return x


def replace_inf_np(x):
    if np.any(np.isinf(x)):
        x[np.isinf(x)] = 2e+300
        x[np.isnan(x)] = 2e+300
    return x


def replace_nan_theano(p):
    if tt.any(tt.isnan(p)):
        p = tt.switch(tt.isnan(p), 1.0, p)
        p = tt.switch(tt.isinf(p), 1.0, p)
        p = normalize(p)
    return p


def replace_nan_np(p):
    if np.any(np.isnan(p)):
        p[np.isnan(p)] = 1.0
        p[np.isinf(p)] = 1.0
        p = normalize(p)
    return p


def logsumexpnp(x, axis=1):
    max_x = x.max(axis=axis, keepdims=True)
    x = x - max_x
    f = np.squeeze(max_x, axis=axis) + np.log(np.sum(np.exp(x), axis=axis))
    return np.expand_dims(f, axis=axis)


def logsumexptheano(x, axis=1):
    max_x = x.max(axis=axis, keepdims=True)
    x = x - max_x
    f = max_x.squeeze() + tt.log(tt.sum(tt.exp(x), axis=axis))
    if axis == 1:
        f = f[:, None]
    if axis == 2:
        f = f[:, :, None]
    if axis == 3:
        f = f[:, :, :, None]
    return f


def softmax_theano(x, axis=1):
    """
        Take softmax for the given two dimensional numpy array.
        :param axis: The axis around which the softmax is applied
        :param x: array-like, shape (n_samples, n_objects)
        :return: softmax taken around the axis=1
    """
    lse = logsumexptheano(x, axis=axis)
    return tt.exp(x - lse)


def softmax_np(x, axis=1):
    """
        Take softmax for the given two dimensional numpy array.
        :param axis: The axis around which the softmax is applied
        :param x: array-like, shape (n_samples, n_objects)
        :return: softmax taken around the axis=1
    """
    lse = logsumexpnp(x, axis=axis)
    return np.exp(x - lse)
