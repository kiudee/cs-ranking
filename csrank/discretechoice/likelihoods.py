import theano
import theano.tensor as tt
from pymc3 import Discrete
from pymc3.distributions.dist_math import bound

from csrank.theano_util import normalize


def categorical_crossentropy(p, y_true):
    return -tt.nnet.categorical_crossentropy(p, y_true)


def binary_crossentropy(p, y_true):
    if p.ndim > 1:
        l = tt.nnet.binary_crossentropy(p, y_true).mean(axis=1)
    else:
        l = tt.nnet.binary_crossentropy(p, y_true).mean(axis=0)
    return -l


def categorical_hinge(p, y_true):
    pos = tt.sum(y_true * p, axis=-1)
    neg = tt.max((1. - y_true) * p, axis=-1)
    return -tt.maximum(0., neg - pos + 1.)


likelihood_dict = {'categorical_crossentropy': categorical_crossentropy, 'binary_crossentropy': binary_crossentropy,
                   'categorical_hinge': categorical_hinge}


class LogLikelihood(Discrete):
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

    def __init__(self, loss_func, p, *args, **kwargs):
        super(LogLikelihood, self).__init__(*args, **kwargs)
        if loss_func is None:
            loss_func = categorical_crossentropy
        self.loss_func = loss_func
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
        p = normalize(p)
        sum_to1 = theano.gradient.zero_grad(
            tt.le(abs(tt.sum(p, axis=-1) - 1), 1e-5))

        value_k = tt.argmax(value, axis=1)
        return bound(a, value_k >= 0, value_k <= (k - 1), sum_to1)
