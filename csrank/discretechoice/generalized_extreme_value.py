import logging
from itertools import product

import numpy as np
import pymc3 as pm
import theano
import theano.tensor as tt
from sklearn.utils import check_random_state

import csrank.numpy_util as npu
import csrank.theano_util as ttu
from csrank.learner import Learner
from csrank.util import print_dictionary
from .discrete_choice import DiscreteObjectChooser
from .likelihoods import likelihood_dict, LogLikelihood


class GeneralizedExtremeValueModel(DiscreteObjectChooser, Learner):

    def __init__(self, n_object_features, n_objects, n_nests=None, loss_function='None', alpha=5e-2, random_state=None,
                 **kwd):
        self.n_object_features = n_object_features
        self.n_objects = n_objects
        if n_nests is None:
            self.n_nests = n_objects + int(n_objects / 2)
        else:
            self.n_nests = n_nests
        self.alpha = alpha
        self.random_state = check_random_state(random_state)
        self.logger = logging.getLogger(GeneralizedExtremeValueModel.__name__)
        self.loss_function = likelihood_dict.get(loss_function, None)
        self.model = None
        self.trace = None
        self.trace_vi = None
        self.Xt = None
        self.Yt = None
        self.p = None

    def get_probabilities(self, utility, lambda_k, alpha_ik):
        n_nests = self.n_nests
        n_instances, n_objects = utility.shape
        pik = tt.zeros((n_instances, n_objects, n_nests))
        sum_per_nest = tt.zeros((n_instances, n_nests))
        for i in range(n_nests):
            uti = (utility + tt.log(alpha_ik[:, :, i])) * 1 / lambda_k[i]
            sum_n = ttu.logsumexp(uti)
            pik = tt.set_subtensor(pik[:, :, i], tt.exp(uti - sum_n))
            sum_per_nest = tt.set_subtensor(sum_per_nest[:, i], sum_n[:, 0] * lambda_k[i])
        pnk = tt.exp(sum_per_nest - ttu.logsumexp(sum_per_nest))
        pnk = pnk[:, None, :]
        p = pik * pnk
        p = p.sum(axis=2)
        return p

    def get_probabilities_np(self, utility, lambda_k, alpha_ik):
        n_nests = self.n_nests
        n_instances, n_objects = utility.shape
        pik = np.zeros((n_instances, n_objects, n_nests))
        sum_per_nest_x = np.zeros((n_instances, n_nests))
        for i in range(n_nests):
            uti = (utility + np.log(alpha_ik[:, :, i])) * 1 / lambda_k[i]
            sum_n = npu.logsumexp(uti)
            pik[:, :, i] = np.exp(uti - sum_n)
            sum_per_nest_x[:, i] = sum_n[:, 0] * lambda_k[i]
        pnk = np.exp(sum_per_nest_x - npu.logsumexp(sum_per_nest_x))
        pnk = pnk[:, None, :]
        p = pik * pnk
        p = p.sum(axis=2)
        return p

    def fit(self, X, Y, sampler="vi", **kwargs):

        with pm.Model() as self.model:
            self.Xt = theano.shared(X)
            self.Yt = theano.shared(Y)
            mu_weights = pm.Normal('mu_weights', mu=0., sd=10)
            sigma_weights = pm.HalfCauchy('sigma_weights', beta=1)
            weights = pm.Normal('weights', mu=mu_weights, sd=sigma_weights, shape=self.n_object_features)

            mu_weights_k = pm.Normal('mu_weights_k', mu=0., sd=10)
            sigma_weights_k = pm.HalfCauchy('sigma_weights_k', beta=1)
            weights_ik = pm.Normal('weights_ik', mu=mu_weights_k, sd=sigma_weights_k,
                                   shape=(self.n_object_features, self.n_nests))

            alpha_ik = tt.dot(self.Xt, weights_ik)
            alpha_ik = ttu.softmax(alpha_ik, axis=2)
            utility = tt.dot(self.Xt, weights)
            lambda_k = pm.Uniform('lambda_k', self.alpha, 1.0, shape=self.n_nests)
            n_instances, n_objects, n_features = X.shape
            self.p = self.get_probabilities(utility, lambda_k, alpha_ik, n_instances, n_objects)
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
        mean_trace = dict(pm.summary(self.trace)['mean'])
        weights = np.array([mean_trace['weights__{}'.format(i)] for i in range(self.n_object_features)])
        lambda_k = np.array([mean_trace['lambda_k__{}'.format(i)] for i in range(self.n_nests)])
        weights_ik = np.zeros((self.n_object_features, self.n_nests))
        for i, k in product(range(self.n_object_features), range(self.n_nests)):
            weights_ik[i][k] = mean_trace['weights_ik__{}_{}'.format(i, k)]
        alpha_ik = np.dot(X, weights_ik)
        alpha_ik = npu.softmax(alpha_ik, axis=2)
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

    def set_tunable_parameters(self, alpha=5e-2, n_nests=None, loss_function='', **point):
        self.alpha = alpha
        if n_nests is None:
            self.n_nests = self.n_objects + int(self.n_objects / 2)
        else:
            self.n_nests = n_nests
        self.loss_function = likelihood_dict.get(loss_function, None)
        if len(point) > 0:
            self.logger.warning('This ranking algorithm does not support tunable parameters'
                                ' called: {}'.format(print_dictionary(point)))
