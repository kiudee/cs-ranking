import logging

from csrank.discretechoice.discrete_choice import DiscreteObjectChooser
from csrank.tunable import Tunable
import theano.tensor as tt
import pymc3 as pm
import numpy as np

from csrank.util import print_dictionary

class MultinomialLogitModel(DiscreteObjectChooser, Tunable):
    def __init__(self, n_features, n_tune=500, n_sample=500):
        self.n_tune = n_tune
        self.n_sample = n_sample
        self.n_features = n_features
        self.logger = logging.getLogger(MultinomialLogitModel.__name__)

    def fit(self, X, Y, **kwargs):
        with pm.Model() as self.model:
            mu_weights = pm.Normal('mu_weights', mu=0., sd=10)
            sigma_weights = pm.HalfCauchy('sigma_weights', beta=1)
            weights = pm.Normal('weights', mu=mu_weights, sd=sigma_weights, shape=self.n_features)
            intercept = pm.Normal('intercept', mu=0, sd=10)
            utility = pm.math.sum(weights * X, axis=2) + intercept
            p = tt.nnet.sigmoid(utility)
            yl = pm.Categorical('yl', p=p, observed=Y)
            self.trace = pm.sample(self.n_sample, tune=self.n_tune, cores=8)

    def _predict_scores_fixed(self, X, **kwargs):
        d = dict(pm.summary(self.trace)['mean'])
        intercept = 0.0
        weights = np.array([d['weights__{}'.format(i)] for i in range(self.n_features)])
        if 'intercept' in d:
            intercept = intercept + d['intercept']
        return np.sum(X * weights, axis=2) + intercept

    def predict(self, X, **kwargs):
        return super().predict(X, **kwargs)

    def predict_scores(self, X, **kwargs):
        return super().predict_scores(X, **kwargs)

    def set_tunable_parameters(self, n_tune=500, n_sample=500, **point):
        self.n_tune = n_tune
        self.n_sample = n_sample
        if len(point) > 0:
            self.logger.warning('This ranking algorithm does not support'
                                ' tunable parameters'
                                ' called: {}'.format(print_dictionary(point)))

