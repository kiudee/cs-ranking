import copy
import logging

from csrank.theano_util import normalize

try:
    import pymc3 as pm
    from pymc3 import Discrete
    from pymc3.distributions.dist_math import bound
    from pymc3.variational.callbacks import CheckParametersConvergence
except ImportError:
    from csrank.util import MissingExtraError

    raise MissingExtraError("pymc3", "probabilistic")

try:
    import theano
    from theano import tensor as tt
except ImportError:
    from csrank.util import MissingExtraError

    raise MissingExtraError("theano", "probabilistic")
logger = logging.getLogger(__name__)


def categorical_crossentropy(p, y_true):
    return -tt.nnet.categorical_crossentropy(p, y_true)


def binary_crossentropy(p, y_true):
    axis = 1 if p.ndim > 1 else 0
    return -tt.nnet.binary_crossentropy(p, y_true).mean(axis=axis)


def categorical_hinge(p, y_true):
    pos = tt.sum(y_true * p, axis=-1)
    neg = tt.max((1.0 - y_true) * p, axis=-1)
    return -tt.maximum(0.0, neg - pos + 1.0)


likelihood_dict = {
    "categorical_crossentropy": categorical_crossentropy,
    "binary_crossentropy": binary_crossentropy,
    "categorical_hinge": categorical_hinge,
}


class LogLikelihood(Discrete):
    """
    Categorical log-likelihood.

    The most general discrete distribution.

    .. math:: f(x \\mid p) = p_x

    ========  ===================================
    Support   :math:`x \\in \\{0, 1, \\ldots, |p|-1\\}`
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
        sum_to1 = theano.gradient.zero_grad(tt.le(abs(tt.sum(p, axis=-1) - 1), 1e-5))

        value_k = tt.argmax(value, axis=1)
        return bound(a, value_k >= 0, value_k <= (k - 1), sum_to1)


def create_weight_dictionary(model_args, shapes):
    weights_dict = dict()
    for key, value in model_args.items():
        prior, params = copy.deepcopy(value)
        for k in params.keys():
            if isinstance(params[k], tuple):
                params[k][1]["name"] = "{}_{}".format(key, k)
                params[k] = params[k][0](**params[k][1])
        params["name"] = key
        params["shape"] = shapes[key]
        weights_dict[key] = prior(**params)
    return weights_dict


def fit_pymc3_model(self, sampler, draws, tune, vi_params, **kwargs):
    callbacks = vi_params.get("callbacks", [])
    for i, c in enumerate(callbacks):
        if isinstance(c, CheckParametersConvergence):
            params = c.__dict__
            params.pop("_diff")
            params.pop("prev")
            params.pop("ord")
            params["diff"] = "absolute"
            callbacks[i] = CheckParametersConvergence(**params)
    if sampler == "variational":
        with self.model:
            try:
                self.trace_ = pm.sample(chains=2, cores=8, tune=5, draws=5)
                vi_params["start"] = self.trace_[-1]
                self.trace_vi_ = pm.fit(**vi_params)
                self.trace_ = self.trace_vi_.sample(draws=draws)
            except Exception as e:
                if hasattr(e, "message"):
                    message = e.message
                else:
                    message = e
                logger.error(message)
                self.trace_vi_ = None
        if self.trace_vi_ is None and self.trace_ is None:
            with self.model:
                logger.info(
                    "Error in vi ADVI sampler using Metropolis sampler with draws {}".format(
                        draws
                    )
                )
                self.trace = pm.sample(
                    chains=1, cores=4, tune=20, draws=20, step=pm.NUTS()
                )
    elif sampler == "metropolis":
        with self.model:
            start = pm.find_MAP()
            self.trace_ = pm.sample(
                chains=2,
                cores=8,
                tune=tune,
                draws=draws,
                **kwargs,
                step=pm.Metropolis(),
                start=start,
            )
    else:
        with self.model:
            self.trace_ = pm.sample(
                chains=2, cores=8, tune=tune, draws=draws, **kwargs, step=pm.NUTS()
            )
