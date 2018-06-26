import logging
import pickle as pk
from abc import ABCMeta
from itertools import product

import pymc3 as pm

from csrank.util import print_dictionary


class ModelSelector(metaclass=ABCMeta):
    def __init__(self, learner_cls, parameter_keys, model_params, model_path, **kwargs):
        self.priors = [[pm.Normal, {'mu': 0, 'sd': 10}], [pm.Laplace, {'mu': 0, 'b': 10}],
                       [pm.Uniform, {'lower': -100, 'upper': 100}]]
        self.parameter_f = [(pm.Normal, {'mu': 0, 'sd': 5}), (pm.Cauchy, {'alpha': 0, 'beta': 1}), 0, -5, 5]
        self.parameter_s = [(pm.HalfCauchy, {'beta': 2}), (pm.HalfNormal, {'sd': 0.5}), (pm.Exponential, {'lam': 0.5}),
                            (pm.Uniform, {'lower': 1, 'upper': 10}), 10]
        # ,(pm.HalfCauchy, {'beta': 1}), (pm.HalfNormal, {'sd': 1}),(pm.Exponential, {'lam': 1.0})]
        self.learner_cls = learner_cls
        self.model_params = model_params
        self.parameter_keys = parameter_keys
        self.parameters = list(product(self.parameter_f, self.parameter_s))
        self.model_path = model_path
        self.models = dict()
        self.logger = logging.getLogger(ModelSelector.__name__)

    def fit(self, X, Y, **fit_params):
        if len(self.parameter_keys) == 2:
            for p1, p2 in product(self.priors, self.priors):
                for param in self.parameters:
                    self.logger.info("Priors {}, {}".format(p1, p2))
                    self.logger.info("mu sd {}".format(param))
                    model_args = dict()
                    k1 = list(p1[1].keys())
                    k2 = list(p2[1].keys())
                    if p1[0].__name__ != 'Uniform':
                        p1[1] = dict(zip(k1, param))
                    if p2[0].__name__ != 'Uniform':
                        p2[1] = dict(zip(k2, param))
                    model_args[self.parameter_keys[0]] = p1
                    model_args[self.parameter_keys[1]] = p2
                    self.model_params['model_args'] = model_args
                    learner = self.learner_cls(**self.model_params)
                    learner.fit(X, Y, **fit_params)
                    self.models[str((p1, p2, param))] = learner
                    self.logger.info("Model done for priors ")
                f = open(self.model_path, "wb")
                pk.dump(self.models, f)
                f.close()
        else:
            for p in self.priors:
                for param in self.parameters:
                    self.logger.info("Priors {}".format(p))
                    self.logger.info("mu sd {}".format(param))
                    model_args = dict()
                    k1 = list(p[1].keys())
                    if p[0].__name__ != 'Uniform':
                        p[1] = dict(zip(k1, param))
                    model_args[self.parameter_keys[0]] = p
                    self.model_params['model_args'] = model_args
                    learner = self.learner_cls(**self.model_params)
                    learner.fit(X, Y, **fit_params)
                    self.models[str((p, param))] = learner
                    self.logger.info("Model done for priors")
                f = open(self.model_path, "wb")
                pk.dump(self.models, f)
                f.close()
