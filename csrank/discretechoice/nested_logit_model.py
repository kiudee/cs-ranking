import logging

import numpy as np
import pymc3 as pm
import theano
import theano.tensor as tt
from sklearn.cluster import MiniBatchKMeans as clustering
from sklearn.utils import check_random_state

from csrank.tunable import Tunable
from csrank.util import print_dictionary
from .discrete_choice import DiscreteObjectChooser
from .likelihoods import likelihood_dict, LogLikelihood


class NestedLogitModel(DiscreteObjectChooser, Tunable):
    def __init__(self, n_features, n_objects, loss_function='', n_tune=500, n_sample=1000, alpha=1e-3,
                 random_state=None,
                 **kwd):
        self.n_tune = n_tune
        self.n_sample = n_sample
        self.n_features = n_features
        self.n_objects = n_objects
        self.n_nests = int(self.n_objects / 2) + 1
        self.alpha = alpha
        self.random_state = check_random_state(random_state)
        self.logger = logging.getLogger(NestedLogitModel.__name__)
        self.cluster_model = None
        self.features_nests = None
        self.loss_function = likelihood_dict.get(loss_function, None)

    def create_nests(self, X):
        n, n_obj, n_dim = X.shape
        objects = X.reshape(n * n_obj, n_dim)
        if self.cluster_model is None:
            self.cluster_model = clustering(n_clusters=self.n_nests, random_state=self.random_state).fit(objects)
            self.cluster_model.fit(objects)
            self.features_nests = self.cluster_model.cluster_centers_
            prediction = self.cluster_model.labels_
        else:
            prediction = self.cluster_model.fit_predict(objects)
        y_nests = []
        for i in np.arange(0, n * n_obj, step=n_obj):
            nest_ids = prediction[i:i + n_obj]
            y_nests.append(nest_ids)
        y_nests = np.array(y_nests)
        return y_nests

    def fit(self, X, Y, **kwargs):
        y_nests = self.create_nests(X)

        def eval_utility(X_train, y_nests, weights):
            weights_t = theano.shared(np.zeros(tuple(X_train.shape)))
            n_nests = int(np.max(y_nests) + 1)
            for i in range(n_nests):
                rows, cols = np.where(y_nests == i)
                weights_t = tt.set_subtensor(weights_t[rows, cols], weights[i])
            utility = tt.sum(X_train * weights_t, axis=2)
            return utility

        def get_probability(y_nests, utility, lambda_k, utility_k):
            utility = tt.exp(utility)
            n_inst = y_nests.shape[0]
            n_nests = int(np.max(y_nests) + 1)
            ivm = np.zeros((n_inst, n_nests))
            IVM = theano.shared(ivm)
            trans_IVM = theano.shared(np.zeros(tuple(y_nests.shape)))
            j = list(range(n_inst))
            for i in range(n_nests):
                (rows, cols) = np.where(y_nests != i)
                sub_tensor = tt.set_subtensor(utility[rows, cols], 0)
                it = pm.math.sum(sub_tensor, axis=1) + 1.0
                IVM = tt.set_subtensor(IVM[j, [i] * n_inst], it)
            IVM = tt.log(IVM)
            for i in range(n_nests):
                rows, cols = np.where(y_nests != i)
                x_i = tt.exp(((lambda_k[i] - 1) * IVM[:, i]) + utility_k[i])
                sub_tensor = theano.shared(np.ones(tuple(y_nests.shape)))
                sub_tensor = sub_tensor * x_i[:, None]
                sub_tensor = tt.set_subtensor(sub_tensor[rows, cols], 0)
                trans_IVM = trans_IVM + sub_tensor
            denominator = pm.math.sum(tt.exp((IVM * lambda_k) + utility_k), axis=1)
            pr_j = utility * trans_IVM / denominator[:, None]
            return pr_j

        with pm.Model() as self.model:
            mu_weights = pm.Normal('mu_weights', mu=0., sd=10)
            sigma_weights = pm.HalfCauchy('sigma_weights', beta=1)
            weights = pm.Normal('weights', mu=mu_weights, sd=sigma_weights, shape=self.n_features)

            mu_weights_k = pm.Normal('mu_weights_k', mu=0., sd=10)
            sigma_weights_k = pm.HalfCauchy('sigma_weights_k', beta=1)
            weights_k = pm.Normal('weights_k', mu=mu_weights_k, sd=sigma_weights_k, shape=self.n_features)
            lambda_k = pm.Uniform('lambda_k', self.alpha, 1.0 + self.alpha, shape=self.n_nests)
            weights = (weights / lambda_k[:, None])

            utility = eval_utility(X, y_nests, weights)
            utility_k = pm.math.dot(self.features_nests, weights_k)
            p = get_probability(y_nests, utility, lambda_k, utility_k)

            if self.loss_function is None:
                Y = np.argmax(Y, axis=1)
                yl = pm.Categorical('yl', p=p, observed=Y)
                self.trace = pm.sample(self.n_sample, tune=self.n_tune, cores=8)
            else:
                yl = LogLikelihood('yl', loss_func=self.loss_function, p=p, observed=Y)
                self.trace = pm.sample(self.n_sample, tune=self.n_tune, cores=8)

    def _predict_scores_fixed(self, X, **kwargs):
        y_nests = self.create_nests(X)

        def eval_utility(X, y_nests, weights):
            weights_t = np.zeros(tuple(X.shape))
            n_nests = int(np.max(y_nests) + 1)
            for i in range(n_nests):
                rows, cols = np.where(y_nests == i)
                weights_t[rows, cols] = weights[i]
            utility = np.sum(X * weights_t, axis=2)
            return utility

        def get_probability(y_nests, utility, lambda_k, utility_k):
            utility = np.exp(utility)
            n_inst = y_nests.shape[0]
            n_nests = int(np.max(y_nests) + 1)
            IVM = np.zeros((n_inst, n_nests))
            j = list(range(n_inst))
            for i in range(n_nests):
                sub_tensor = np.copy(utility)
                rows, cols = np.where(y_nests != i)
                sub_tensor[rows, cols] = 0
                it = np.log(np.sum(sub_tensor, axis=1) + 1.0)
                IVM[j, [i] * n_inst] = it
            trans_IVM = np.zeros_like(utility)
            for i in range(n_nests):
                rows, cols = np.where(y_nests != i)
                x_i = np.exp((lambda_k[i] - 1) * IVM[:, i] + utility_k[i])  # Wnl here
                sub_tensor = np.ones(tuple(trans_IVM.shape)) * x_i[:, None]
                sub_tensor[rows, cols] = 0
                trans_IVM = trans_IVM + sub_tensor
            denominator = np.sum(np.exp(((IVM * lambda_k) + utility_k)), axis=1)  # Wnl here
            pr_j = utility * trans_IVM / denominator[:, None]
            return pr_j

        d = dict(pm.summary(self.trace)['mean'])
        weights = np.array([d['weights__{}'.format(i)] for i in range(self.n_features)])
        weights_k = np.array([d['weights_k__{}'.format(i)] for i in range(self.n_features)])
        lambda_k = np.array([d['lambda_k__{}'.format(i)] for i in range(self.n_nests)])
        weights = (weights / lambda_k[:, None])

        utility_k = np.dot(self.features_nests, weights_k)
        utility = eval_utility(X, y_nests, weights)
        scores = get_probability(y_nests, utility, lambda_k, utility_k)
        return scores

    def predict(self, X, **kwargs):
        return super().predict(X, **kwargs)

    def predict_scores(self, X, **kwargs):
        return super().predict_scores(X, **kwargs)

    def set_tunable_parameters(self, n_tune=500, n_sample=500, alpha=1e-3, **point):
        self.n_tune = n_tune
        self.n_sample = n_sample
        self.alpha = alpha
        if len(point) > 0:
            self.logger.warning('This ranking algorithm does not support tunable parameters'
                                ' called: {}'.format(print_dictionary(point)))
