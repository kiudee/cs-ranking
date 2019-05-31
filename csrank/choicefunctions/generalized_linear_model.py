import copy
import logging

import numpy as np
import pymc3 as pm
import theano
import theano.tensor as tt
from pymc3 import Discrete
from pymc3.distributions.dist_math import bound
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state

import csrank.theano_util as ttu
from csrank.learner import Learner
from csrank.util import print_dictionary
from .choice_functions import ChoiceFunctions


class GeneralizedLinearModel(ChoiceFunctions, Learner):
    def __init__(self, n_object_features, regularization='l2', random_state=None, **kwargs):
        """
            Create an instance of the GeneralizedLinearModel model.

            Parameters
            ----------
            n_object_features : int
                Number of features of the object space
            regularization : string, optional
                Regularization technique to be used for estimating the weights
            random_state : int or object
                Numpy random state
            **kwargs
                Keyword arguments for the algorithms

            References
            ----------
                [1] Kenneth E Train. „Discrete choice methods with simulation“. In: Cambridge university press, 2009. Chap Logit, pp. 41–86.
        """
        self.logger = logging.getLogger(GeneralizedLinearModel.__name__)
        self.n_object_features = n_object_features
        if regularization in ['l1', 'l2']:
            self.regularization = regularization
        else:
            self.regularization = 'l2'
        self.random_state = check_random_state(random_state)
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
            'weights': [weight, {'mu': (pm.Normal, {'mu': 0, 'sd': 10}), prior: (pm.HalfCauchy, {'beta': 1})}]}
        self.logger.info('Creating default config {}'.format(print_dictionary(config_dict)))
        return config_dict

    def construct_model(self, X, Y):
        self.logger.info('Creating model_args config {}'.format(print_dictionary(self.default_configuration)))
        with pm.Model() as self.model:
            self.Xt = theano.shared(X)
            self.Yt = theano.shared(Y)
            shapes = {'weights': self.n_object_features}
            # shapes = {'weights': (self.n_object_features, 3)}
            weights_dict = create_weight_dictionary(self.default_configuration, shapes)
            intercept = pm.Normal('intercept', mu=0, sd=10)
            utility = tt.dot(self.Xt, weights_dict['weights']) + intercept
            self.p = ttu.sigmoid(utility)
            yl = BinaryCrossEntropyLikelihood('yl', p=self.p, observed=self.Yt)
        self.logger.info("Model construction completed")

    def fit(self, X, Y, sampler='vi', tune_size=0.1, thin_thresholds=1, **kwargs):
        if tune_size > 0:
            X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=tune_size, random_state=self.random_state)
            try:
                self._fit(X_train, Y_train, sampler=sampler, **kwargs)
            finally:
                self.logger.info('Fitting utility function finished. Start tuning threshold.')
                self.threshold = self._tune_threshold(X_val, Y_val, thin_thresholds=thin_thresholds)
        else:
            self._fit(X, Y, sampler=sampler, **kwargs)
            self.threshold = 0.5

    def _fit(self, X, Y, sampler='vi', **kwargs):
        self.construct_model(X, Y)
        callbacks = kwargs['vi_params'].get('callbacks', [])
        kwargs['random_seed'] = self.random_state.randint(2 ** 32, dtype='uint32')

        for i, c in enumerate(callbacks):
            if isinstance(c, pm.callbacks.CheckParametersConvergence):
                params = c.__dict__
                params.pop('_diff')
                params.pop('prev')
                params.pop('ord')
                params['diff'] = 'absolute'
                callbacks[i] = pm.callbacks.CheckParametersConvergence(**params)
        if sampler == 'vi':
            with self.model:
                sample_params = kwargs['sample_params']
                vi_params = kwargs['vi_params']
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
        return ChoiceFunctions.predict_for_scores(self, scores, **kwargs)

    def set_tunable_parameters(self, regularization="l1", **point):
        self.regularization = regularization
        self.model = None
        self.trace = None
        self.trace_vi = None
        self.Xt = None
        self.Yt = None
        self.p = None
        if len(point) > 0:
            self.logger.warning('This ranking algorithm does not support'
                                ' tunable parameters called: {}'.format(print_dictionary(point)))


def create_weight_dictionary(model_args, shapes):
    weights_dict = dict()
    for key, value in model_args.items():
        prior, params = copy.deepcopy(value)
        for k in params.keys():
            if isinstance(params[k], tuple):
                params[k][1]['name'] = '{}_{}'.format(key, k)
                params[k] = params[k][0](**params[k][1])
        params['name'] = key
        params['shape'] = shapes[key]
        weights_dict[key] = prior(**params)
    return weights_dict


def binary_crossentropy(p, y_true):
    if p.ndim > 1:
        l = (tt.nnet.binary_crossentropy(p, y_true).sum(axis=1)).mean()
    else:
        l = tt.nnet.binary_crossentropy(p, y_true).mean(axis=0)
    return -l


def categorical_crossentropy(p, y_true):
    return -tt.nnet.categorical_crossentropy(p, y_true)


def categorical_hinge(p, y_true):
    pos = tt.sum(y_true * p, axis=-1)
    neg = tt.max((1. - y_true) * p, axis=-1)
    return -tt.maximum(0., neg - pos + 1.)


class BinaryCrossEntropyLikelihood(Discrete):
    R"""
    Categorical log-likelihood.

    The most general discrete distribution.

    .. math:: f(x \mid p) = p_x

    ========  ===================================
    Support   :math:`x \in \{0, 1, \ldots, |p|-1\}`
    ========  ===================================

    Parameters
    ----------
    p : array of floats
        p > 0 and the elements of p must sum to 1. They will be automatically
        rescaled otherwise.
    """

    def __init__(self, p, *args, **kwargs):
        super(BinaryCrossEntropyLikelihood, self).__init__(*args, **kwargs)
        self.loss_func = categorical_hinge
        try:
            self.k = tt.shape(p)[-1].tag.test_value
        except AttributeError:
            self.k = tt.shape(p)[-1]
        self.p = tt.as_tensor_variable(p)
        self.mode = tt.argmax(p)

    def random(self, **kwargs):
        return NotImplemented

    def logp(self, value):
        p = self.p
        k = self.k
        a = self.loss_func(p, value)
        p = ttu.normalize(p)
        sum_to1 = theano.gradient.zero_grad(
            tt.le(abs(tt.sum(p, axis=-1) - 1), 1e-5))

        value_k = tt.argmax(value, axis=1)
        return bound(a, value_k >= 0, value_k <= (k - 1), sum_to1)
