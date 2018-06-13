import logging

import numpy as np
import pymc3 as pm
import theano.tensor as tt
from sklearn.cluster import MiniBatchKMeans as clustering
from sklearn.utils import check_random_state

from csrank.discretechoice.util import logsumexpnp, logsumexptheano
from csrank.learner import Learner
from csrank.util import print_dictionary
from .discrete_choice import DiscreteObjectChooser
from .likelihoods import likelihood_dict, LogLikelihood


class NestedLogitModel(DiscreteObjectChooser, Learner):
    def __init__(self, n_object_features, n_objects, n_nests=None, loss_function='', n_tune=500, n_sample=500,
                 alpha=1e-2, random_state=None, **kwd):
        self.n_tune = n_tune
        self.n_sample = n_sample
        self.n_object_features = n_object_features
        self.n_objects = n_objects
        if n_nests is None:
            self.n_nests = int(n_objects / 2)
        else:
            self.n_nests = n_nests
        self.alpha = alpha
        self.random_state = check_random_state(random_state)
        self.logger = logging.getLogger(NestedLogitModel.__name__)
        self.cluster_model = None
        self.features_nests = None
        self.loss_function = likelihood_dict.get(loss_function, None)
        self.model = None
        self.trace = None
        self.trace_vi = None

    def create_nests(self, X):
        n, n_obj, n_dim = X.shape
        objects = X.reshape(n * n_obj, n_dim)
        if self.cluster_model is None:
            self.cluster_model = clustering(n_clusters=self.n_nests, random_state=self.random_state).fit(objects)
            self.cluster_model.fit(objects)
            self.features_nests = self.cluster_model.cluster_centers_
            prediction = self.cluster_model.labels_
        else:
            prediction = self.cluster_model.predict(objects)
        y_nests = []
        for i in np.arange(0, n * n_obj, step=n_obj):
            nest_ids = prediction[i:i + n_obj]
            y_nests.append(nest_ids)
        y_nests = np.array(y_nests)
        return y_nests

    def fit(self, X, Y, sampler="advi", n=20000, cores=8, sample=3, **kwargs):
        y_nests = self.create_nests(X)

        with pm.Model() as self.model:
            mu_weights = pm.Normal('mu_weights', mu=0., sd=10)
            sigma_weights = pm.HalfCauchy('sigma_weights', beta=1)
            weights = pm.Normal('weights', mu=mu_weights, sd=sigma_weights, shape=self.n_object_features)

            mu_weights_k = pm.Normal('mu_weights_k', mu=0., sd=10)
            sigma_weights_k = pm.HalfCauchy('sigma_weights_k', beta=1)
            weights_k = pm.Normal('weights_k', mu=mu_weights_k, sd=sigma_weights_k, shape=self.n_object_features)
            lambda_k = pm.Uniform('lambda_k', self.alpha, 1.0, shape=self.n_nests)
            weights = (weights / lambda_k[:, None])

            utility = eval_utility(X, y_nests, weights)
            utility_k = tt.dot(self.features_nests, weights_k)
            p = get_probability(y_nests, utility, lambda_k, utility_k)

            if self.loss_function is None:
                Y = np.argmax(Y, axis=1)
                yl = pm.Categorical('yl', p=p, observed=Y)
            else:
                yl = LogLikelihood('yl', loss_func=self.loss_function, p=p, observed=Y)
        if sampler in ['advi', 'fullrank_advi', 'svgd']:
            with self.model:
                self.trace = pm.sample(sample, tune=5, cores=cores)
                self.trace_vi = pm.fit(n=n, start=self.trace[-1], method=sampler)
                self.trace = self.trace_vi.sample(draws=self.n_sample)
        elif sampler == 'metropolis':
            with self.model:
                start = pm.find_MAP()
                self.trace = pm.sample(self.n_sample, tune=self.n_tune, step=pm.Metropolis(), start=start, cores=cores)
        else:
            with self.model:
                self.trace = pm.sample(self.n_sample, tune=self.n_tune, step=pm.NUTS(), cores=cores)

    def _predict_scores_fixed(self, X, **kwargs):
        y_nests = self.create_nests(X)
        mean_trace = dict(pm.summary(self.trace)['mean'])
        weights = np.array([mean_trace['weights__{}'.format(i)] for i in range(self.n_object_features)])
        weights_k = np.array([mean_trace['weights_k__{}'.format(i)] for i in range(self.n_object_features)])
        lambda_k = np.array([mean_trace['lambda_k__{}'.format(i)] for i in range(self.n_nests)])
        weights = (weights / lambda_k[:, None])
        utility_k = np.dot(self.features_nests, weights_k)
        utility = eval_utility_np(X, y_nests, weights)
        scores = get_probability_np(y_nests, utility, lambda_k, utility_k)
        return scores

    def predict(self, X, **kwargs):
        return super().predict(X, **kwargs)

    def predict_scores(self, X, **kwargs):
        return super().predict_scores(X, **kwargs)

    def predict_for_scores(self, scores, **kwargs):
        return DiscreteObjectChooser.predict_for_scores(self, scores, **kwargs)

    def clear_memory(self, **kwargs):
        self.logger.info("Clearing memory")
        pass

    def set_tunable_parameters(self, n_tune=500, n_sample=500, alpha=1e-2, n_nests=None, loss_function='', **point):
        self.n_tune = n_tune
        self.n_sample = n_sample
        self.alpha = alpha
        if n_nests is None:
            self.n_nests = int(self.n_objects / 2)
        else:
            self.n_nests = n_nests
        self.loss_function = likelihood_dict.get(loss_function, None)
        if len(point) > 0:
            self.logger.warning('This ranking algorithm does not support tunable parameters'
                                ' called: {}'.format(print_dictionary(point)))


def eval_utility(x_train, y_nests, weights):
    n_nests = int(np.max(y_nests) + 1)
    utility = tt.zeros(tuple(y_nests.shape))
    for i in range(n_nests):
        rows, cols = np.where(y_nests == i)
        utility = tt.set_subtensor(utility[rows, cols], tt.dot(x_train[rows, cols], weights[i]))
    return utility


def get_probability(y_nests, utility, lambda_k, utility_k):
    n_inst = y_nests.shape[0]
    n_nests = int(np.max(y_nests) + 1)
    pni_k = tt.zeros_like(utility)
    j = list(range(n_inst))
    ivm = tt.zeros((n_inst, n_nests))
    for i in range(n_nests):
        sub_tensor = tt.set_subtensor(utility[np.where(y_nests != i)], -1e50)
        ink = logsumexptheano(sub_tensor)
        pni_k = tt.set_subtensor(pni_k[np.where(y_nests == i)], tt.exp(sub_tensor - ink)[np.where(y_nests == i)])
        ivm = tt.set_subtensor(ivm[:, i], lambda_k[i] * ink[:, 0] + utility_k[i])
    pk = tt.exp(ivm - logsumexptheano(ivm))
    pn_k = tt.zeros_like(pni_k)
    for i in range(n_nests):
        rows, cols = np.where(y_nests == i)
        p = tt.ones_like(pn_k) * pk[:, i][:, None]
        pn_k = tt.set_subtensor(pn_k[rows, cols], p[rows, cols])
    p = pni_k * pn_k
    return p


def eval_utility_np(x_t, y_nests, weights):
    n_nests = int(np.max(y_nests) + 1)
    utility = np.zeros(tuple(y_nests.shape))
    for i in range(n_nests):
        rows, cols = np.where(y_nests == i)
        utility[rows, cols] = np.dot(x_t[rows, cols], weights[i])
    return utility


def get_probability_np(y_nests, utility, lambda_k, utility_k):
    n_inst = y_nests.shape[0]
    n_nests = int(np.max(y_nests) + 1)
    pni_k = np.zeros_like(utility)
    j = list(range(n_inst))
    ivm = np.zeros((n_inst, n_nests))
    for i in range(n_nests):
        sub_tensor = np.copy(utility)
        sub_tensor[np.where(y_nests != i)] = -1e50
        ink = logsumexpnp(sub_tensor)
        pni_k[np.where(y_nests == i)] = np.exp(sub_tensor - ink)[np.where(y_nests == i)]
        ivm[:, i] = lambda_k[i] * ink[:, 0] + utility_k[i]
    pk = np.exp(ivm - logsumexpnp(ivm))
    pn_k = np.zeros_like(pni_k)
    for i in range(n_nests):
        rows, cols = np.where(y_nests == i)
        p = (np.ones(tuple(pn_k.shape)) * pk[:, i][:, None])
        pn_k[rows, cols] = p[rows, cols]
    p = pni_k * pn_k
    return p
