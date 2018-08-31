import copy
import logging

import numpy as np
import pymc3 as pm
import theano
import theano.tensor as tt
from sklearn.cluster import MiniBatchKMeans
from sklearn.utils import check_random_state

import csrank.numpy_util as npu
import csrank.theano_util as ttu
from csrank.discretechoice.likelihoods import create_weight_dictionary
from csrank.learner import Learner
from csrank.util import print_dictionary
from .discrete_choice import DiscreteObjectChooser
from .likelihoods import likelihood_dict, LogLikelihood


class NestedLogitModel(DiscreteObjectChooser, Learner):
    def __init__(self, n_object_features, n_objects, n_nests=None, loss_function='', regularization='l1', alpha=1e-2,
                 random_state=None, model_args={}, **kwd):
        self.logger = logging.getLogger(NestedLogitModel.__name__)
        self.n_object_features = n_object_features
        self.n_objects = n_objects
        if n_nests is None:
            self.n_nests = int(n_objects / 2)
        else:
            self.n_nests = n_nests
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
        self.cluster_model = None
        self.features_nests = None
        self.model = None
        self.trace = None
        self.trace_vi = None
        self.Xt = None
        self.Yt = None
        self.p = None
        self.y_nests = None

    @property
    def default_configuration(self):
        if self.regularization == 'l2':
            weight = pm.Normal
            prior = 'sd'
        elif self.regularization == 'l1':
            weight = pm.Laplace
            prior = 'b'
        config_dict = {
            'weights': [weight, {'mu': (pm.Normal, {'mu': 0, 'sd': 5}), prior: (pm.HalfCauchy, {'beta': 1})}],
            'weights_k': [weight, {'mu': (pm.Normal, {'mu': 0, 'sd': 5}), prior: (pm.HalfCauchy, {'beta': 1})}]}
        self.logger.info('Creating default config {}'.format(print_dictionary(config_dict)))

        return config_dict

    def eval_utility(self, weights):
        utility = tt.zeros(tuple(self.y_nests.shape))
        for i in range(self.n_nests):
            rows, cols = tt.eq(self.y_nests, i).nonzero()
            utility = tt.set_subtensor(utility[rows, cols], tt.dot(self.Xt[rows, cols], weights[i]))
        return utility

    def get_probability(self, utility, lambda_k, utility_k):
        n_instances, n_objects = self.y_nests.shape
        pni_k = tt.zeros((n_instances, n_objects))
        ivm = tt.zeros((n_instances, self.n_nests))
        for i in range(self.n_nests):
            rows, cols = tt.neq(self.y_nests, i).nonzero()
            sub_tensor = tt.set_subtensor(utility[rows, cols], -1e50)
            ink = ttu.logsumexp(sub_tensor)
            rows, cols = tt.eq(self.y_nests, i).nonzero()
            pni_k = tt.set_subtensor(pni_k[rows, cols], tt.exp(sub_tensor - ink)[rows, cols])
            ivm = tt.set_subtensor(ivm[:, i], lambda_k[i] * ink[:, 0] + utility_k[i])
        pk = tt.exp(ivm - ttu.logsumexp(ivm))
        pn_k = tt.zeros((n_instances, n_objects))
        for i in range(self.n_nests):
            rows, cols = tt.eq(self.y_nests, i).nonzero()
            p = tt.ones((n_instances, n_objects)) * pk[:, i][:, None]
            pn_k = tt.set_subtensor(pn_k[rows, cols], p[rows, cols])
        p = pni_k * pn_k
        return p

    def eval_utility_np(self, x_t, y_nests, weights):
        utility = np.zeros(tuple(y_nests.shape))
        for i in range(self.n_nests):
            rows, cols = np.where(y_nests == i)
            utility[rows, cols] = np.dot(x_t[rows, cols], weights[i])
        return utility

    def get_probability_np(self, y_nests, utility, lambda_k, utility_k):
        n_instances, n_objects = y_nests.shape
        pni_k = np.zeros((n_instances, n_objects))
        ivm = np.zeros((n_instances, self.n_nests))
        for i in range(self.n_nests):
            sub_tensor = np.copy(utility)
            sub_tensor[np.where(y_nests != i)] = -1e50
            ink = npu.logsumexp(sub_tensor)
            pni_k[np.where(y_nests == i)] = np.exp(sub_tensor - ink)[np.where(y_nests == i)]
            ivm[:, i] = lambda_k[i] * ink[:, 0] + utility_k[i]
        pk = np.exp(ivm - npu.logsumexp(ivm))
        pn_k = np.zeros((n_instances, n_objects))
        for i in range(self.n_nests):
            rows, cols = np.where(y_nests == i)
            p = np.ones((n_instances, n_objects)) * pk[:, i][:, None]
            pn_k[rows, cols] = p[rows, cols]
        p = pni_k * pn_k
        return p

    def create_nests(self, X):
        n, n_obj, n_dim = X.shape
        objects = X.reshape(n * n_obj, n_dim)
        if self.cluster_model is None:
            self.cluster_model = MiniBatchKMeans(n_clusters=self.n_nests, random_state=self.random_state).fit(objects)
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

    def construct_model(self, X, Y):
        for key, value in self.default_configuration.items():
            self.model_args[key] = self.model_args.get(key, value)
        self.logger.info('Creating model_args config {}'.format(print_dictionary(self.model_args)))
        y_nests = self.create_nests(X)
        with pm.Model() as self.model:
            self.Xt = theano.shared(X)
            self.Yt = theano.shared(Y)
            self.y_nests = theano.shared(y_nests)
            shapes = {'weights': self.n_object_features, 'weights_k': self.n_object_features}

            weights_dict = create_weight_dictionary(self.model_args, shapes)
            lambda_k = pm.Uniform('lambda_k', self.alpha, 1.0, shape=self.n_nests)
            weights = (weights_dict['weights'] / lambda_k[:, None])
            utility = self.eval_utility(weights)
            utility_k = tt.dot(self.features_nests, weights_dict['weights_k'])
            self.p = self.get_probability(utility, lambda_k, utility_k)

            yl = LogLikelihood('yl', loss_func=self.loss_function, p=self.p, observed=self.Yt)
        self.logger.info("Model construction completed")

    def fit(self, X, Y, sampler="vi", **kwargs):
        self.construct_model(X, Y)
        callbacks = kwargs['vi_params'].get('callbacks', [])
        for i, c in enumerate(callbacks):
            if isinstance(c, pm.callbacks.CheckParametersConvergence):
                params = c.__dict__
                params.pop('_diff')
                params.pop('prev')
                params.pop('ord')
                params['diff'] = 'absolute'
                print(params)
                callbacks[i] = pm.callbacks.CheckParametersConvergence(**params)
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
        y_nests = self.create_nests(X)
        mean_trace = dict(pm.summary(self.trace)['mean'])
        weights = np.array([mean_trace['weights__{}'.format(i)] for i in range(self.n_object_features)])
        weights_k = np.array([mean_trace['weights_k__{}'.format(i)] for i in range(self.n_object_features)])
        lambda_k = np.array([mean_trace['lambda_k__{}'.format(i)] for i in range(self.n_nests)])
        weights = (weights / lambda_k[:, None])
        utility_k = np.dot(self.features_nests, weights_k)
        utility = self.eval_utility_np(X, y_nests, weights)
        scores = self.get_probability_np(y_nests, utility, lambda_k, utility_k)
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

    def set_tunable_parameters(self, alpha=None, n_nests=None, loss_function='', regularization="l1", **point):
        if alpha is not None:
            self.alpha = alpha
        if n_nests is None:
            self.n_nests = int(self.n_objects / 2)
        else:
            self.n_nests = n_nests
        self.regularization = regularization
        if loss_function in likelihood_dict.keys():
            self.loss_function = likelihood_dict.get(loss_function, None)
        self.cluster_model = None
        self.features_nests = None
        self.model = None
        self.trace = None
        self.trace_vi = None
        self.Xt = None
        self.Yt = None
        self.p = None
        self.y_nests = None
        self.model_args = dict()

        if len(point) > 0:
            self.logger.warning('This ranking algorithm does not support tunable parameters'
                                ' called: {}'.format(print_dictionary(point)))
