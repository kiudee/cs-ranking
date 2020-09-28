from abc import ABCMeta
from itertools import product
import logging
import pickle as pk

import numpy as np

try:
    import pymc3 as pm
except ImportError:
    from csrank.util import MissingExtraError

    raise MissingExtraError("pymc3", "probabilistic")

logger = logging.getLogger(__name__)


class ModelSelector(metaclass=ABCMeta):
    def __init__(
        self,
        learner_cls,
        parameter_keys,
        model_params,
        fit_params,
        model_path,
        **kwargs,
    ):
        self.priors = [
            [pm.Normal, {"mu": 0, "sd": 10}],
            [pm.Laplace, {"mu": 0, "b": 10}],
        ]
        self.uniform_prior = [pm.Uniform, {"lower": -20, "upper": 20}]
        self.prior_indices = np.arange(len(self.priors))
        self.parameter_f = [
            (pm.Normal, {"mu": 0, "sd": 5}),
            (pm.Cauchy, {"alpha": 0, "beta": 1}),
            0,
            -5,
            5,
        ]
        self.parameter_s = [
            (pm.HalfCauchy, {"beta": 1}),
            (pm.HalfNormal, {"sd": 0.5}),
            (pm.Exponential, {"lam": 0.5}),
            (pm.Uniform, {"lower": 1, "upper": 10}),
            10,
        ]
        # ,(pm.HalfCauchy, {'beta': 2}), (pm.HalfNormal, {'sd': 1}),(pm.Exponential, {'lam': 1.0})]
        self.learner_cls = learner_cls
        self.model_params = model_params
        self.fit_params = fit_params
        self.parameter_keys = parameter_keys
        self.parameters = list(product(self.parameter_f, self.parameter_s))
        pf_arange = np.arange(len(self.parameter_f))
        ps_arange = np.arange(len(self.parameter_s))
        self.parameter_ind = list(product(pf_arange, ps_arange))
        self.model_path = model_path
        self.models = dict()

    def fit(self, X, Y):
        self._pre_fit()
        model_args = dict()
        for param_key in self.parameter_keys:
            model_args[param_key] = self.uniform_prior
        logger.info("Uniform Prior")
        self.model_params["model_args"] = model_args
        key = "{}_uniform_prior".format(self.parameter_keys)
        self.fit_learner(X, Y, key)
        for j, param in enumerate(self.parameters):
            logger.info("mu: {}, sd/b: {}".format(*self.parameter_ind[j]))
            if len(self.parameter_keys) == 2:
                for i1, i2 in product(self.prior_indices, self.prior_indices):
                    prior1 = self.priors[i1]
                    prior2 = self.priors[i2]
                    logger.info("Priors {}, {}".format(i1, i2))
                    model_args = dict()
                    k1 = list(prior1[1].keys())
                    k2 = list(prior2[1].keys())
                    prior1[1] = dict(zip(k1, param))
                    prior2[1] = dict(zip(k2, param))
                    model_args[self.parameter_keys[0]] = prior1
                    model_args[self.parameter_keys[1]] = prior2
                    key = "{}_{}_{}_{}_mu_{}_sd_{}".format(
                        self.parameter_keys[0],
                        i1,
                        self.parameter_keys[1],
                        i2,
                        self.parameter_ind[j][0],
                        self.parameter_ind[j][1],
                    )
                    self.model_params["model_args"] = model_args
                    self.fit_learner(X, Y, key)
            else:
                for i, prior in enumerate(self.priors):
                    logger.info("Prior {}".format(i))
                    model_args = dict()
                    k1 = list(prior[1].keys())
                    prior[1] = dict(zip(k1, param))
                    model_args[self.parameter_keys[0]] = prior
                    self.model_params["model_args"] = model_args
                    key = "{}_{}_mu_{}_sd_{}".format(
                        self.parameter_keys[0],
                        i,
                        self.parameter_ind[j][0],
                        self.parameter_ind[j][1],
                    )
                    self.fit_learner(X, Y, key)
        return self

    def fit_learner(self, X, Y, key):
        learner = self.learner_cls(**self.model_params)
        try:
            learner.fit(X, Y, **self.fit_params)
            self.models[key] = {"model": learner.model, "trace": learner.trace}
            logger.info("Model done for priors key {}".format(key))
            f = open(self.model_path, "wb")
            pk.dump(self.models, f)
            f.close()
        except Exception as e:
            logger.error("Error for parameters {}: {}".format(key, e))
