import theano
import theano.tensor as tt
from pymc3 import Categorical
from pymc3.distributions.dist_math import bound


def categorical_crossentropy(p, y_true):
    return -tt.nnet.categorical_crossentropy(p, y_true)


def binary_crossentropy(p, y_true):
    if p.ndim > 1:
        l = tt.nnet.binary_crossentropy(p, y_true).mean(axis=1)
    else:
        l = tt.nnet.binary_crossentropy(p, y_true).mean(axis=0)
    return l


def categorical_hinge(p, y_true):
    pos = tt.sum(y_true * p, axis=-1)
    neg = tt.max((1. - y_true) * p, axis=-1)
    return -1 * tt.maximum(0., neg - pos + 1.)


likelihood_dict = {'categorical_crossentropy': categorical_crossentropy, 'binary_crossentropy': binary_crossentropy,
                   'categorical_hinge': categorical_hinge}


class LogLikelihood(Categorical):
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

    def __init__(self, loss_func, *args, **kwargs):
        super(LogLikelihood, self).__init__(*args, **kwargs)
        self.loss_func = loss_func

    def random(self, point=None, size=None, repeat=None):
        return super().random(point, size, repeat)

    def logp(self, value):
        p = self.p
        # Clip values before using them for indexing
        sumto1value = theano.gradient.zero_grad(
            tt.le(abs(tt.sum(value, axis=-1) - 1), 1e-5))
        sumto1 = theano.gradient.zero_grad(
            tt.le(abs(tt.sum(p, axis=-1) - 1), 1e-5))
        a = self.loss_func(p, value)
        return bound(a, sumto1value, sumto1)
