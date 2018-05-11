import logging

import numpy as np
import pymc3 as pm
import theano
import theano.tensor as tt
from sklearn.preprocessing import LabelBinarizer

from csrank.discretechoice.discrete_choice import DiscreteObjectChooser
from csrank.discretechoice.likelihoods import LogLikelihood, likelihood_dict
from csrank.tunable import Tunable
from csrank.util import print_dictionary


class MultinomialLogitModel(DiscreteObjectChooser, Tunable):
    def __init__(self, n_features, n_tune=500, n_sample=500, loss_function='', **kwargs):
        self.n_tune = n_tune
        self.n_sample = n_sample
        self.n_features = n_features
        self.logger = logging.getLogger(MultinomialLogitModel.__name__)
        self.loss_function = likelihood_dict.get(loss_function, None)

    def fit(self, X, Y, loss_func=None, **kwargs):
        with pm.Model() as self.model:
            mu_weights = pm.Normal('mu_weights', mu=0., sd=10)
            sigma_weights = pm.HalfCauchy('sigma_weights', beta=1)
            weights = pm.Normal('weights', mu=mu_weights, sd=sigma_weights, shape=self.n_features)
            intercept = pm.Normal('intercept', mu=0, sd=10)
            utility = pm.math.sum(weights * X, axis=2) + intercept
            p = tt.nnet.sigmoid(utility)
            if self.loss_function is None:
                yl = pm.Categorical('yl', p=p, observed=Y)
                self.trace = pm.sample(self.n_sample, tune=self.n_tune, cores=8)
            else:
                Y = LabelBinarizer().fit_transform(Y)
                Y = theano.shared(Y)
                yl = LogLikelihood('yl', loss_func=self.loss_function, p=p, observed=Y)
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
