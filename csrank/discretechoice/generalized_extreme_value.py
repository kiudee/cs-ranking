import logging
from itertools import product

import numpy as np
import pymc3 as pm
import theano.tensor as tt
from sklearn.utils import check_random_state

from csrank.learner import Learner
from csrank.util import print_dictionary, softmax
from .discrete_choice import DiscreteObjectChooser
from .likelihoods import likelihood_dict, LogLikelihood


class GeneralizedExtremeValueModel(DiscreteObjectChooser, Learner):

    def __init__(self, n_object_features, n_objects, n_nests=None, loss_function='None', n_tune=500, n_sample=1000,
                 alpha=1e-3, beta=2.0, random_state=None, **kwd):
        self.n_tune = n_tune
        self.n_sample = n_sample
        self.n_object_features = n_object_features
        self.n_objects = n_objects
        if n_nests is None:
            self.n_nests = n_objects + int(n_objects / 2)
        else:
            self.n_nests = n_nests
        self.alpha = alpha
        self.beta = beta
        self.random_state = check_random_state(random_state)
        self.logger = logging.getLogger(GeneralizedExtremeValueModel.__name__)
        self.loss_function = likelihood_dict.get(loss_function, None)
        self.model = None
        self.trace = None

    def get_probabilities(self, utility, lambda_k, alpha_ik, n_instances, n_objects):
        utility = tt.exp(utility)
        utility_nest = tt.zeros((n_instances, n_objects, self.n_nests))
        for i in range(self.n_nests):
            uti = tt.power(utility * alpha_ik[:, :, i], 1 / lambda_k[i])
            utility_nest = tt.set_subtensor(utility_nest[:, :, i], uti)
        sum_per_nest = utility_nest.sum(axis=1)
        denominators = tt.sum(tt.power(sum_per_nest, lambda_k), axis=1)
        p = tt.zeros_like(utility)
        for i in range(n_objects):
            numerator_i = tt.power(sum_per_nest, (lambda_k - 1)) * utility_nest[:, i, :]
            p = tt.set_subtensor(p[:, i], numerator_i.sum(axis=1) / denominators)
        return p

    def get_probabilities_np(self, utility, lambda_k, alpha_ik):
        n_instances, n_objects = utility.shape
        utility = np.exp(utility)
        utility_nest = np.zeros((n_instances, n_objects, self.n_nests))
        for i in range(self.n_nests):
            uti = np.power(utility * alpha_ik[:, :, i], 1 / lambda_k[i])
            utility_nest[:, :, i] = uti

        sum_per_nest = utility_nest.sum(axis=1)
        denominators = np.sum(np.power(sum_per_nest, lambda_k), axis=1)
        p = np.zeros_like(utility)
        for i in range(n_objects):
            numerator_i = np.power(sum_per_nest, (lambda_k - 1)) * utility_nest[:, i, :]
            p[:, i] = numerator_i.sum(axis=1) / denominators
        return p

    def fit(self, X, Y, **kwargs):
        with pm.Model() as self.model:
            mu_weights = pm.Normal('mu_weights', mu=0., sd=10)
            sigma_weights = pm.HalfCauchy('sigma_weights', beta=1)
            weights = pm.Normal('weights', mu=mu_weights, sd=sigma_weights, shape=self.n_object_features)

            mu_weights_k = pm.Normal('mu_weights_k', mu=0., sd=10)
            sigma_weights_k = pm.HalfCauchy('sigma_weights_k', beta=1)
            weights_ik = pm.Normal('weights_ik', mu=mu_weights_k, sd=sigma_weights_k,
                                   shape=(self.n_object_features, self.n_nests))
            alpha_ik = tt.dot(X, weights_ik)
            alpha_ik = softmax(alpha_ik, axis=2)
            utility = tt.dot(X, weights)
            lambda_k = pm.HalfCauchy('lambda_k', beta=self.beta, shape=self.n_nests) + self.alpha
            # lambda_k = pm.Uniform('lambda_k', self.alpha, self.beta + self.alpha, shape=self.n_nests)
            n_instances, n_objects, n_features = X.shape
            p = self.get_probabilities(utility, lambda_k, alpha_ik, n_instances, n_objects)

            if self.loss_function is None:
                Y = np.argmax(Y, axis=1)
                yl = pm.Categorical('yl', p=p, observed=Y)
                self.trace = pm.sample(self.n_sample, tune=self.n_tune, cores=8)
            else:
                yl = LogLikelihood('yl', loss_func=self.loss_function, p=p, observed=Y)
                self.trace = pm.sample(self.n_sample, tune=self.n_tune, cores=8)

    def _predict_scores_fixed(self, X, **kwargs):
        mean_trace = dict(pm.summary(self.trace)['mean'])
        weights = np.array([mean_trace['weights__{}'.format(i)] for i in range(self.n_object_features)])
        lambda_k = np.array([mean_trace['lambda_k__{}'.format(i)] for i in range(self.n_nests)])
        weights_ik = np.zeros((self.n_object_features, self.n_nests))
        for i, k in product(range(self.n_object_features), range(self.n_nests)):
            weights_ik[i][k] = mean_trace['weights_ik__{}_{}'.format(i, k)]
        alpha_ik = np.dot(X, weights_ik)
        alpha_ik = softmax(alpha_ik, axis=2)
        utility = np.dot(X, weights)
        p = self.get_probabilities_np(utility, lambda_k, alpha_ik)
        return p

    def predict(self, X, **kwargs):
        return super().predict(X, **kwargs)

    def predict_scores(self, X, **kwargs):
        return super().predict_scores(X, **kwargs)

    def predict_for_scores(self, scores, **kwargs):
        return DiscreteObjectChooser.predict_for_scores(self, scores, **kwargs)

    def clear_memory(self, **kwargs):
        self.logger.info("Clearing memory")
        pass

    def set_tunable_parameters(self, n_tune=500, n_sample=1000, alpha=1e-3, beta=2.0, n_nests=None, loss_function='',
                               **point):
        self.n_tune = n_tune
        self.n_sample = n_sample
        self.alpha = alpha
        self.beta = beta
        if n_nests is None:
            self.n_nests = self.n_objects + int(self.n_objects / 2)
        else:
            self.n_nests = n_nests
        self.loss_function = likelihood_dict.get(loss_function, None)
        if len(point) > 0:
            self.logger.warning('This ranking algorithm does not support tunable parameters'
                                ' called: {}'.format(print_dictionary(point)))
