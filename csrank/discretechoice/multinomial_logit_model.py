import logging

import numpy as np
import pymc3 as pm
import theano
import theano.tensor as tt

import csrank.theano_util as ttu
from csrank.learner import Learner
from csrank.util import print_dictionary
from .discrete_choice import DiscreteObjectChooser
from .likelihoods import likelihood_dict, LogLikelihood, create_weight_dictionary

default_configuration = {
    'weights': (pm.Normal, {'mu': (pm.Normal, {'mu': 0, 'sd': 10}), 'sd': (pm.HalfCauchy, {'beta': 2})})}


class MultinomialLogitModel(DiscreteObjectChooser, Learner):
    def __init__(self, n_object_features, loss_function='', model_args={}, **kwargs):
        self.n_object_features = n_object_features
        self.logger = logging.getLogger(MultinomialLogitModel.__name__)
        self.loss_function = likelihood_dict.get(loss_function, None)
        self.model_args = dict()
        for key, value in default_model_configuration.items():
            self.model_args[key] = model_args.get(key, value)

        self.model = None
        self.trace = None
        self.trace_vi = None
        self.Xt = None
        self.Yt = None
        self.p = None
        self.mu_weights = None

    def construct_model(self, X, Y):
        with pm.Model() as self.model:
            self.Xt = theano.shared(X)
            self.Yt = theano.shared(Y)
            shapes = {'weights': self.n_object_features}

            weights_dict = create_weight_dictionary(self.model_args, shapes)
            intercept = pm.Normal('intercept', mu=0, sd=10)
            utility = tt.dot(self.Xt, weights_dict['weights']) + intercept
            self.p = ttu.softmax(utility, axis=1)

            yl = LogLikelihood('yl', loss_func=self.loss_function, p=self.p, observed=self.Yt)

    def fit(self, X, Y, sampler='vi', **kwargs):
        self.construct_model(X, Y)
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
        d = dict(pm.summary(self.trace)['mean'])
        intercept = 0.0
        weights = np.array([d['weights__{}'.format(i)] for i in range(self.n_object_features)])
        if 'intercept' in d:
            intercept = intercept + d['intercept']
        return np.dot(X, weights) + intercept

    def predict(self, X, **kwargs):
        return super().predict(X, **kwargs)

    def predict_scores(self, X, **kwargs):
        return super().predict_scores(X, **kwargs)

    def predict_for_scores(self, scores, **kwargs):
        return DiscreteObjectChooser.predict_for_scores(self, scores, **kwargs)

    def clear_memory(self, **kwargs):
        self.logger.info("Clearing memory")
        pass

    def set_tunable_parameters(self, loss_function='', **point):
        self.loss_function = likelihood_dict.get(loss_function, None)
        if len(point) > 0:
            self.logger.warning('This ranking algorithm does not support'
                                ' tunable parameters called: {}'.format(print_dictionary(point)))
