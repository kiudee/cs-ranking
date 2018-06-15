import logging
from itertools import combinations

import numpy as np
import pymc3 as pm
import theano.tensor as tt
from sklearn.utils import check_random_state

from csrank.discretechoice.util import logsumexptheano, logsumexpnp
from csrank.learner import Learner
from csrank.util import print_dictionary
from .discrete_choice import DiscreteObjectChooser
from .likelihoods import likelihood_dict, LogLikelihood


class PairedCombinatorialLogit(DiscreteObjectChooser, Learner):

    def __init__(self, n_object_features, n_objects, loss_function='', n_tune=500, n_sample=500, alpha=5e-2,
                 random_state=None, **kwd):
        self.logger = logging.getLogger(PairedCombinatorialLogit.__name__)
        self.n_tune = n_tune
        self.n_sample = n_sample
        self.n_object_features = n_object_features
        self.n_objects = n_objects
        self.nests_indices = np.array(list(combinations(np.arange(n_objects), 2)))
        self.n_nests = len(self.nests_indices)
        self.alpha = alpha
        self.random_state = check_random_state(random_state)
        self.loss_function = likelihood_dict.get(loss_function, None)
        self.model = None
        self.trace = None
        self.trace_vi = None

    def get_probabilities(self, utility, lambda_k, n_instances):
        n_objects = self.n_objects
        nests_indices = self.nests_indices
        n_nests = self.n_nests
        lambdas = tt.ones((n_objects, n_objects), dtype=np.float)
        for i, p in enumerate(nests_indices):
            r = [p[0], p[1]]
            c = [p[1], p[0]]
            lambdas = tt.set_subtensor(lambdas[r, c], lambda_k[i])
        uti_per_nest = tt.transpose(utility[:, None, :] / lambdas, (0, 2, 1))
        ind = np.array([[[i1, i2], [i2, i1]] for i1, i2 in nests_indices])
        ind = ind.reshape(2 * n_nests, 2)
        x = uti_per_nest[:, ind[:, 0], ind[:, 1]].reshape((n_instances * n_nests, 2))
        log_sum_exp_nest = logsumexptheano(x).reshape((n_instances, n_nests))
        pnk = tt.exp(log_sum_exp_nest * lambda_k - logsumexptheano(log_sum_exp_nest * lambda_k))
        p = tt.zeros((n_instances, n_objects), dtype=float)
        for i in range(n_nests):
            i1, i2 = nests_indices[i]
            x1 = tt.exp(uti_per_nest[:, i1, i2] - log_sum_exp_nest[:, i]) * pnk[:, i]
            x2 = np.exp(uti_per_nest[:, i2, i1] - log_sum_exp_nest[:, i]) * pnk[:, i]
            p = tt.set_subtensor(p[:, i1], p[:, i1] + x1)
            p = tt.set_subtensor(p[:, i2], p[:, i2] + x2)
        return p

    def get_probabilities_np(self, utility, lambda_k):
        n_instances = utility.shape[0]
        n_objects = self.n_objects
        nests_indices = self.nests_indices
        n_nests = self.n_nests
        lambdas = np.ones((n_objects, n_objects), lambda_k.dtype)
        lambdas[nests_indices[:, 0], nests_indices[:, 1]] = lambdas.T[
            nests_indices[:, 0], nests_indices[:, 1]] = lambda_k
        uti_per_nest = np.transpose((utility[:, None] / lambdas), (0, 2, 1))
        ind = np.array([[[i1, i2], [i2, i1]] for i1, i2 in nests_indices])
        ind = ind.reshape(2 * n_nests, 2)
        x = uti_per_nest[:, ind[:, 0], ind[:, 1]].reshape(n_instances * n_nests, 2)
        log_sum_exp_nest = logsumexpnp(x).reshape(n_instances, n_nests)
        pnk = np.exp(log_sum_exp_nest * lambda_k - logsumexpnp(log_sum_exp_nest * lambda_k))
        p = np.zeros((n_instances, n_objects), dtype=float)
        for i in range(n_nests):
            i1, i2 = nests_indices[i]
            p[:, i1] += np.exp(uti_per_nest[:, i1, i2] - log_sum_exp_nest[:, i]) * pnk[:, i]
            p[:, i2] += np.exp(uti_per_nest[:, i2, i1] - log_sum_exp_nest[:, i]) * pnk[:, i]
        return p

    def fit(self, X, Y, sampler="advi", n=20000, **kwargs):
        with pm.Model() as self.model:
            mu_weights = pm.Normal('mu_weights', mu=0., sd=10)
            sigma_weights = pm.HalfCauchy('sigma_weights', beta=1)
            weights = pm.Normal('weights', mu=mu_weights, sd=sigma_weights, shape=self.n_object_features)
            # weights = pm.Uniform('weights', lower=-1.0, upper=1.0, shape=self.n_object_features)
            utility = tt.dot(X, weights)

            # Numerical stability
            utility = utility - (utility.max(axis=1, keepdims=True) + utility.min(axis=1, keepdims=True)) / 2

            lambda_k = pm.Uniform('lambda_k', self.alpha, 1.0, shape=self.n_nests)
            # lambda_k = pm.HalfCauchy('lambda_k', beta=2, shape=self.n_nests) + self.alpha
            # lambda_k = lambda_k/lambda_k.sum()
            # lambda_k = pm.Uniform('lambda_k', self.alpha, 1.0)
            # lambda_k = tt.zeros(self.n_nests) + lambda_k
            p = self.get_probabilities(utility, lambda_k, X.shape[0])

            if self.loss_function is None:
                Y = np.argmax(Y, axis=1)
                yl = pm.Categorical('yl', p=p, observed=Y)
            else:
                yl = LogLikelihood('yl', loss_func=self.loss_function, p=p, observed=Y)
        if sampler in ['advi', 'fullrank_advi', 'svgd']:
            with self.model:
                self.trace = pm.sample(5, tune=5, cores=8)
                self.trace_vi = pm.fit(n=n, start=self.trace[-1], method=sampler)
                self.trace = self.trace_vi.sample(draws=self.n_sample)
        elif sampler == 'metropolis':
            with self.model:
                start = pm.find_MAP()
                self.trace = pm.sample(self.n_sample, tune=self.n_tune, step=pm.Metropolis(), start=start, cores=8)
        else:
            with self.model:
                self.trace = pm.sample(self.n_sample, tune=self.n_tune, step=pm.NUTS(), cores=8)

    def _predict_scores_fixed(self, X, **kwargs):
        mean_trace = dict(pm.summary(self.trace)['mean'])
        weights = np.array([mean_trace['weights__{}'.format(i)] for i in range(self.n_object_features)])
        lambda_k = np.array([mean_trace['lambda_k__{}'.format(i)] for i in range(self.n_nests)])
        utility = np.dot(X, weights)
        p = self.get_probabilities_np(utility, lambda_k)
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

    def set_tunable_parameters(self, n_tune=500, n_sample=500, alpha=5e-2, loss_function='', **point):
        self.n_tune = n_tune
        self.n_sample = n_sample
        self.alpha = alpha
        self.loss_function = likelihood_dict.get(loss_function, None)
        if len(point) > 0:
            self.logger.warning('This ranking algorithm does not support tunable parameters'
                                ' called: {}'.format(print_dictionary(point)))
