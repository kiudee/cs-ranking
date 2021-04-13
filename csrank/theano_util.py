try:
    from theano import tensor as tt
except ImportError:
    from csrank.util import MissingExtraError

    raise MissingExtraError("theano", "probabilistic")


def replace_inf_theano(x):
    if tt.any(tt.isinf(x)):
        x = tt.switch(tt.isinf(x), 2e300, x)
        x = tt.switch(tt.isnan(x), 2e300, x)
    return x


def replace_nan_theano(p):
    if tt.any(tt.isnan(p)):
        p = tt.switch(tt.isnan(p), 1.0, p)
        p = tt.switch(tt.isinf(p), 1.0, p)
        p = normalize(p)
    return p


def logsumexp(x, axis=1):
    x_max = tt.max(x, axis=axis, keepdims=True)
    return tt.log(tt.sum(tt.exp(x - x_max), axis=axis, keepdims=True)) + x_max


def softmax(x, axis=1):
    """
    Take softmax for the given two dimensional theano array.
    :param axis: The axis around which the softmax is applied
    :param x: array-like, shape (n_samples, n_objects)
    :return: softmax taken around the axis=1
    """
    lse = logsumexp(x, axis=axis)
    return tt.exp(x - lse)


def sigmoid(x):
    x = 1 / (1 + tt.exp(-1 * x))
    return x


def normalize(x, axis=1):
    """
    Normalize the given two dimensional theano array around the row.
    :param axis: The axis around which the norm is applied
    :param x: theano or numpy array-like, shape (n_samples, n_objects)
    :return: normalize the array around the axis=1
    """
    return x / tt.sum(x, axis=axis, keepdims=True)
