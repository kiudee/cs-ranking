import logging

import numpy as np
import pymc3 as pm
import theano
import theano.tensor as tt
from sklearn.cluster import MiniBatchKMeans as clustering
from sklearn.utils import check_random_state

from csrank.learner import Learner
import csrank.numpy_util as npu
import csrank.theano_util as ttu
from csrank.util import print_dictionary
from .discrete_choice import DiscreteObjectChooser
from .likelihoods import likelihood_dict, LogLikelihood


class NestedLogitModel(DiscreteObjectChooser, Learner):
    def __init__(self, n_object_features, n_objects, n_nests=None, loss_function='', alpha=1e-2, random_state=None,
                 **kwd):
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
        self.Xt = None
        self.Yt = None
        self.p = None

    def create_nests(self, X):
        n, n_obj, n_dim = X.shape
        objects = X.reshape(n * n_obj, n_dim)
        if self.cluster_model is None:
            self.cluster_model = clustering(n_clusters=self.n_nests, random_state=self.random_state).fit(objects)
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

    def fit(self, X, Y, sampler="vi", **kwargs):
        y_nests = self.create_nests(X)

        with pm.Model() as self.model:
            self.Xt = theano.shared(X)
            self.Yt = theano.shared(Y)
            mu_weights = pm.Normal('mu_weights', mu=0., sd=10)
            sigma_weights = pm.HalfCauchy('sigma_weights', beta=1)
            weights = pm.Normal('weights', mu=mu_weights, sd=sigma_weights, shape=self.n_object_features)

            mu_weights_k = pm.Normal('mu_weights_k', mu=0., sd=10)
            sigma_weights_k = pm.HalfCauchy('sigma_weights_k', beta=1)
            weights_k = pm.Normal('weights_k', mu=mu_weights_k, sd=sigma_weights_k, shape=self.n_object_features)
            lambda_k = pm.Uniform('lambda_k', self.alpha, 1.0, shape=self.n_nests)
            weights = (weights / lambda_k[:, None])

            utility = eval_utility(self.Xt, y_nests, weights)
            utility_k = tt.dot(self.features_nests, weights_k)
            self.p = get_probability(y_nests, utility, lambda_k, utility_k)

            yl = LogLikelihood('yl', loss_func=self.loss_function, p=self.p, observed=self.Yt)

        if sampler == 'vi':
            with self.model:
                sample_params = kwargs['sample_params']
                self.trace = pm.sample(**sample_params)
                vi_params = kwargs['vi_params']
                vi_params['start'] = self.trace[-1]
                self.trace_vi = pm.fit(**vi_params)
                self.trace = self.trace_vi.sample(draws=kwargs['draws'])
        elif sampler == 'metropolis':
            with self.model:
                start = pm.find_MAP()
                self.trace = pm.sample(**kwargs, step=pm.Metropolis(), start=start)
        else:
            with self.model:
                self.trace = pm.sample(**kwargs, step=pm.NUTS())

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

    def set_tunable_parameters(self, alpha=1e-2, n_nests=None, loss_function='', **point):
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
    n_instances, n_objects = y_nests.shape
    n_nests = int(np.max(y_nests) + 1)
    pni_k = tt.zeros((n_instances, n_objects))
    ivm = tt.zeros((n_instances, n_nests))
    for i in range(n_nests):
        sub_tensor = tt.set_subtensor(utility[np.where(y_nests != i)], -1e50)
        ink = ttu.logsumexp(sub_tensor)
        pni_k = tt.set_subtensor(pni_k[np.where(y_nests == i)], tt.exp(sub_tensor - ink)[np.where(y_nests == i)])
        ivm = tt.set_subtensor(ivm[:, i], lambda_k[i] * ink[:, 0] + utility_k[i])
    pk = tt.exp(ivm - ttu.logsumexp(ivm))
    pn_k = tt.zeros((n_instances, n_objects))
    for i in range(n_nests):
        rows, cols = np.where(y_nests == i)
        p = tt.ones((n_instances, n_objects)) * pk[:, i][:, None]
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
    n_instances, n_objects = y_nests.shape
    n_nests = int(np.max(y_nests) + 1)
    pni_k = np.zeros((n_instances, n_objects))
    ivm = np.zeros((n_instances, n_nests))
    for i in range(n_nests):
        sub_tensor = np.copy(utility)
        sub_tensor[np.where(y_nests != i)] = -1e50
        ink = npu.logsumexp(sub_tensor)
        pni_k[np.where(y_nests == i)] = np.exp(sub_tensor - ink)[np.where(y_nests == i)]
        ivm[:, i] = lambda_k[i] * ink[:, 0] + utility_k[i]
    pk = np.exp(ivm - npu.logsumexp(ivm))
    pn_k = np.zeros((n_instances, n_objects))
    for i in range(n_nests):
        rows, cols = np.where(y_nests == i)
        p = np.ones((n_instances, n_objects)) * pk[:, i][:, None]
        pn_k[rows, cols] = p[rows, cols]
    p = pni_k * pn_k
    return p
