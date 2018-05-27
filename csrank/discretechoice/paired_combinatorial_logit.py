import logging
from itertools import combinations

import numpy as np
import pymc3 as pm
import theano
import theano.tensor as tt
from sklearn.utils import check_random_state

from csrank.tunable import Tunable
from csrank.util import print_dictionary
from .discrete_choice import DiscreteObjectChooser
from .likelihoods import likelihood_dict, LogLikelihood


class PairedCombinatorialLogit(DiscreteObjectChooser, Tunable):

    def __init__(self, n_features, n_objects, loss_function='', n_tune=500, n_sample=1000, alpha=1e-3,
                 random_state=None,
                 **kwd):
        self.n_tune = n_tune
        self.n_sample = n_sample
        self.n_features = n_features
        self.n_objects = n_objects
        self.nests_indices = np.array(list(combinations(np.arange(n_objects), 2)))
        self.n_nests = len(self.nests_indices)
        self.alpha = alpha
        self.random_state = check_random_state(random_state)
        self.logger = logging.getLogger(PairedCombinatorialLogit.__name__)
        self.loss_function = likelihood_dict.get(loss_function, None)
        self.model = None
        self.trace = None

    def get_probabilities(self, utility, lambda_k, n_instances):
        lambdas = tt.zeros((self.n_objects, self.n_objects), dtype=np.float)
        lambda_k = lambda_k / lambda_k.sum()
        for i, p in enumerate(self.nests_indices):
            r = [p[0], p[1]]
            c = [p[1], p[0]]
            lambdas = tt.set_subtensor(lambdas[r, c], lambda_k[i])
        p = tt.zeros((n_instances, self.n_objects))
        denominators = theano.shared(np.zeros(n_instances))
        for i in range(self.n_objects):
            other = list(range(self.n_objects))
            other.remove(i)
            for j in other:
                x = tt.exp(utility[:, i] / lambdas[i, j]) + tt.exp(utility[:, j] / lambdas[i, j])
                numerator = tt.exp(utility[:, i] / lambdas[i, j]) * tt.power(x, lambdas[i, j] - 1)
                p = tt.set_subtensor(p[:, i], p[:, i] + numerator)
                denominators += tt.power(x, lambdas[i, j])
        denominators = denominators / 2
        p = p / denominators[:, None]
        return p

    def get_probabilities_np(self, utility, lambda_k):
        n_instances = utility.shape[0]
        lambdas = np.zeros((self.n_objects, self.n_objects), dtype=np.float)
        lambda_k = lambda_k / lambda_k.sum()
        lambdas[self.nests_indices[:, 0], self.nests_indices[:, 1]] = lambdas.T[
            self.nests_indices[:, 0], self.nests_indices[:, 1]] = lambda_k

        p = np.zeros((n_instances, self.n_objects))
        denominators = np.zeros(n_instances)
        for i in range(self.n_objects):
            other = list(range(self.n_objects))
            other.remove(i)
            for j in other:
                x = np.exp(utility[:, i] / lambdas[i, j]) + np.exp(utility[:, j] / lambdas[i, j])
                numerator = np.exp(utility[:, i] / lambdas[i, j]) * np.power(x, lambdas[i, j] - 1)
                p[:, i] += numerator
                denominators += np.power(x, lambdas[i, j])
        denominators = denominators / 2
        p = p / denominators[:, None]
        return p

    def fit(self, X, Y, **kwargs):
        with pm.Model() as self.model:
            mu_weights = pm.Normal('mu_weights', mu=0., sd=10)
            sigma_weights = pm.HalfCauchy('sigma_weights', beta=1)
            weights = pm.Normal('weights', mu=mu_weights, sd=sigma_weights, shape=self.n_features)

            utility = pm.math.sum(weights * X, axis=2)
            lambda_k = pm.Uniform('lambda_k', self.alpha, 5.0 + self.alpha, shape=self.n_nests)
            p = self.get_probabilities(utility, lambda_k, X.shape[0])

            if self.loss_function is None:
                Y = np.argmax(Y, axis=1)
                yl = pm.Categorical('yl', p=p, observed=Y)
                self.trace = pm.sample(self.n_sample, tune=self.n_tune, cores=8)
            else:
                yl = LogLikelihood('yl', loss_func=self.loss_function, p=p, observed=Y)
                self.trace = pm.sample(self.n_sample, tune=self.n_tune, cores=8)

    def _predict_scores_fixed(self, X, **kwargs):
        d = dict(pm.summary(self.trace)['mean'])
        weights = np.array([d['weights__{}'.format(i)] for i in range(self.n_features)])
        lambda_k = np.array([d['lambda_k__{}'.format(i)] for i in range(self.n_nests)])
        utility = np.sum(X * weights, axis=2)
        p = self.get_probabilities_np(utility, lambda_k)
        return p

    def predict(self, X, **kwargs):
        return super().predict(X, **kwargs)

    def predict_scores(self, X, **kwargs):
        return super().predict_scores(X, **kwargs)

    def set_tunable_parameters(self, n_tune=500, n_sample=1000, alpha=1e-3, **point):
        self.n_tune = n_tune
        self.n_sample = n_sample
        self.alpha = alpha
        if len(point) > 0:
            self.logger.warning('This ranking algorithm does not support tunable parameters'
                                ' called: {}'.format(print_dictionary(point)))
