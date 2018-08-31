import copy
import logging
from itertools import combinations

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
from .likelihoods import likelihood_dict, LogLikelihood, create_weight_dictionary


class PairedCombinatorialLogit(DiscreteObjectChooser, Learner):

    def __init__(self, n_object_features, n_objects, loss_function='', regularization='l2', alpha=5e-2,
                 random_state=None, model_args={}, **kwd):
        self.logger = logging.getLogger(PairedCombinatorialLogit.__name__)
        self.n_object_features = n_object_features
        self.n_objects = n_objects
        self.nests_indices = np.array(list(combinations(np.arange(n_objects), 2)))
        self.n_nests = len(self.nests_indices)
        self.alpha = alpha
        self.random_state = check_random_state(random_state)
        self.loss_function = likelihood_dict.get(loss_function, None)
        if regularization in ['l1', 'l2']:
            self.regularization = regularization
        else:
            self.regularization = 'l2'
        if isinstance(model_args, dict):
            self.model_args = model_args
        else:
            self.model_args = dict()

        self.model = None
        self.trace = None
        self.trace_vi = None
        self.Xt = None
        self.Yt = None
        self.p = None

    @property
    def default_configuration(self):
        if self.regularization == 'l2':
            weight = pm.Normal
            prior = 'sd'
        elif self.regularization == 'l1':
            weight = pm.Laplace
            prior = 'b'
        config_dict = {
            'weights': [weight, {'mu': (pm.Normal, {'mu': 0, 'sd': 5}), prior: (pm.HalfCauchy, {'beta': 1})}]}
        self.logger.info('Creating default config {}'.format(print_dictionary(config_dict)))
        return config_dict

    def get_probabilities(self, utility, lambda_k):
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
        x = uti_per_nest[:, ind[:, 0], ind[:, 1]].reshape((-1, 2))
        log_sum_exp_nest = ttu.logsumexp(x).reshape((-1, n_nests))
        pnk = tt.exp(log_sum_exp_nest * lambda_k - ttu.logsumexp(log_sum_exp_nest * lambda_k))
        p = tt.zeros(tuple(utility.shape), dtype=float)
        for i in range(n_nests):
            i1, i2 = nests_indices[i]
            x1 = tt.exp(uti_per_nest[:, i1, i2] - log_sum_exp_nest[:, i]) * pnk[:, i]
            x2 = np.exp(uti_per_nest[:, i2, i1] - log_sum_exp_nest[:, i]) * pnk[:, i]
            p = tt.set_subtensor(p[:, i1], p[:, i1] + x1)
            p = tt.set_subtensor(p[:, i2], p[:, i2] + x2)
        return p

    def get_probabilities_np(self, utility, lambda_k):
        n_objects = self.n_objects
        nests_indices = self.nests_indices
        n_nests = self.n_nests
        lambdas = np.ones((n_objects, n_objects), lambda_k.dtype)
        lambdas[nests_indices[:, 0], nests_indices[:, 1]] = lambdas.T[
            nests_indices[:, 0], nests_indices[:, 1]] = lambda_k
        uti_per_nest = np.transpose((utility[:, None] / lambdas), (0, 2, 1))
        ind = np.array([[[i1, i2], [i2, i1]] for i1, i2 in nests_indices])
        ind = ind.reshape(2 * n_nests, 2)
        x = uti_per_nest[:, ind[:, 0], ind[:, 1]].reshape(-1, 2)
        log_sum_exp_nest = npu.logsumexp(x).reshape(-1, n_nests)
        pnk = np.exp(log_sum_exp_nest * lambda_k - npu.logsumexp(log_sum_exp_nest * lambda_k))
        p = np.zeros(tuple(utility.shape), dtype=float)
        for i in range(n_nests):
            i1, i2 = nests_indices[i]
            p[:, i1] += np.exp(uti_per_nest[:, i1, i2] - log_sum_exp_nest[:, i]) * pnk[:, i]
            p[:, i2] += np.exp(uti_per_nest[:, i2, i1] - log_sum_exp_nest[:, i]) * pnk[:, i]
        return p

    def construct_model(self, X, Y):
        for key, value in self.default_configuration.items():
            self.model_args[key] = self.model_args.get(key, value)
        self.logger.info('Creating model_args config {}'.format(print_dictionary(self.model_args)))
        with pm.Model() as self.model:
            self.Xt = theano.shared(X)
            self.Yt = theano.shared(Y)
            shapes = {'weights': self.n_object_features}
            weights_dict = create_weight_dictionary(self.model_args, shapes)
            lambda_k = pm.Uniform('lambda_k', self.alpha, 1.0, shape=self.n_nests)
            utility = tt.dot(self.Xt, weights_dict['weights'])
            self.p = self.get_probabilities(utility, lambda_k)
            yl = LogLikelihood('yl', loss_func=self.loss_function, p=self.p, observed=self.Yt)
        self.logger.info("Model construction completed")

    def fit(self, X, Y, sampler="vi", **kwargs):
        self.construct_model(X, Y)
        if sampler == 'vi':
            random_seed = kwargs['random_seed']
            with self.model:
                sample_params = kwargs['sample_params']
                vi_params = kwargs['vi_params']
                vi_params['random_seed'] = sample_params['random_seed'] = random_seed
                draws_ = kwargs['draws']
                try:
                    self.trace = pm.sample(**sample_params)
                    vi_params['start'] = self.trace[-1]
                    self.trace_vi = pm.fit(**vi_params)
                    self.trace = self.trace_vi.sample(draws=draws_)
                except Exception as e:
                    if hasattr(e, 'message'):
                        message = e.message
                    else:
                        message = e
                    self.logger.error(message)
                    self.trace_vi = None
                    self.trace = None
            if self.trace_vi is None and self.trace is None:
                with self.model:
                    self.logger.info("Error in vi ADVI sampler using nuts sampler with draws {}".format(draws_))
                    nuts_params = copy.deepcopy(sample_params)
                    nuts_params['tune'] = nuts_params['draws'] = 50
                    self.logger.info("Params {}".format(nuts_params))
                    self.trace = pm.sample(**nuts_params)
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

    def set_tunable_parameters(self, alpha=5e-2, loss_function='', regularization='l2', **point):
        if alpha is not None:
            self.alpha = alpha
        if loss_function in likelihood_dict.keys():
            self.loss_function = likelihood_dict.get(loss_function, None)
        self.regularization = regularization
        self.model = None
        self.trace = None
        self.trace_vi = None
        self.Xt = None
        self.Yt = None
        self.p = None
        self.model_args = dict()
        if len(point) > 0:
            self.logger.warning('This ranking algorithm does not support tunable parameters'
                                ' called: {}'.format(print_dictionary(point)))
