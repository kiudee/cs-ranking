import copy
import logging
from itertools import product

import numpy as np
import pymc3 as pm
import theano
import theano.tensor as tt

import csrank.numpy_util as npu
import csrank.theano_util as ttu
from csrank.learner import Learner
from csrank.util import print_dictionary
from .discrete_choice import DiscreteObjectChooser
from .likelihoods import likelihood_dict, LogLikelihood, create_weight_dictionary


class MixedLogitModel(DiscreteObjectChooser, Learner):
    def __init__(self, n_object_features, n_mixtures=4, loss_function='', regularization='l2', model_args={}, **kwargs):
        self.logger = logging.getLogger(MixedLogitModel.__name__)
        self.n_object_features = n_object_features
        self.loss_function = likelihood_dict.get(loss_function, None)
        if regularization in ['l1', 'l2']:
            self.regularization = regularization
        else:
            self.regularization = 'l2'
        if isinstance(model_args, dict):
            self.model_args = model_args
        else:
            self.model_args = dict()
        self.n_mixtures = n_mixtures
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

    def construct_model(self, X, Y):
        for key, value in self.default_configuration.items():
            self.model_args[key] = self.model_args.get(key, value)
        self.logger.info('Creating model_args config {}'.format(print_dictionary(self.model_args)))
        with pm.Model() as self.model:
            self.Xt = theano.shared(X)
            self.Yt = theano.shared(Y)
            shapes = {'weights': (self.n_object_features, self.n_mixtures)}
            weights_dict = create_weight_dictionary(self.model_args, shapes)
            utility = tt.dot(self.Xt, weights_dict['weights'])
            self.p = tt.mean(ttu.softmax(utility, axis=1), axis=2)
            yl = LogLikelihood('yl', loss_func=self.loss_function, p=self.p, observed=self.Yt)
        self.logger.info("Model construction completed")

    def fit(self, X, Y, sampler='vi', **kwargs):
        self.construct_model(X, Y)
        callbacks = kwargs['vi_params'].get('callbacks', [])
        for i, c in enumerate(callbacks):
            if isinstance(c, pm.callbacks.CheckParametersConvergence):
                params = c.__dict__
                params.pop('_diff')
                params.pop('prev')
                params.pop('ord')
                params['diff'] = 'absolute'
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
        summary = dict(pm.summary(self.trace)['mean'])
        weights = np.zeros((self.n_object_features, self.n_mixtures))
        for i, k in product(range(self.n_object_features), range(self.n_mixtures)):
            weights[i][k] = summary['weights__{}_{}'.format(i, k)]
        utility = np.dot(X, weights)
        p = np.mean(npu.softmax(utility, axis=1), axis=2)
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

    def set_tunable_parameters(self, loss_function='', regularization="l1", n_mixtures=4, **point):
        if loss_function in likelihood_dict.keys():
            self.loss_function = likelihood_dict.get(loss_function, None)
        self.n_mixtures = n_mixtures
        self.regularization = regularization
        self.model = None
        self.trace = None
        self.trace_vi = None
        self.Xt = None
        self.Yt = None
        self.p = None
        self.model_args = dict()
        if len(point) > 0:
            self.logger.warning('This ranking algorithm does not support'
                                ' tunable parameters called: {}'.format(print_dictionary(point)))
