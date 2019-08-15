import logging
from itertools import product

import numpy as np
import pymc3 as pm
import theano
import theano.tensor as tt
from pymc3.variational.callbacks import CheckParametersConvergence

import csrank.numpy_util as npu
import csrank.theano_util as ttu
from csrank.learner import Learner
from csrank.util import print_dictionary
from .discrete_choice import DiscreteObjectChooser
from .likelihoods import likelihood_dict, LogLikelihood, create_weight_dictionary, fit_pymc3_model


class MixedLogitModel(DiscreteObjectChooser, Learner):
    def __init__(self, n_object_features, n_mixtures=4, loss_function='', regularization='l2', **kwargs):
        """
            Create an instance of the Mixed Logit model for learning the discrete choice function. In this model we
            assume weights of this model to be random due to which this model can learn different variations in choices
            amongst the individuals. The utility score for each object in query set :math:`Q` is defined as
            :math:`U_r(x) = w_r \cdot x`, where :math:`w_r` is the k-th sample weight vector from the underlying distribution
            The probability of choosing an object :math:`x_i` is defined by taking softmax over the
            utility scores of the objects:

            .. math::

                P(x_i \\lvert Q) = \\frac{1}{R} \sum_{r=1}^R \\frac{exp(U_r(x_i))}{\sum_{x_j \in Q} exp(U_r(x_j))}

            The discrete choice for the given query set :math:`Q` is defined as:

            .. math::

                dc(Q) := \operatorname{argmax}_{x_i \in Q }  \; P(x_i \\lvert Q)

            Parameters
            ----------
            n_object_features : int
                Number of features of the object space
            n_mixtures: int (range : [2, inf])
                The number of logit models (:math:`R`) which are used to estimate the choice probability
            loss_function : string , {‘categorical_crossentropy’, ‘binary_crossentropy’, ’categorical_hinge’}
                Loss function to be used for the discrete choice decision from the query set
            regularization : string, {‘l1’, ‘l2’}, string
               Regularizer function (L1 or L2) applied to the `kernel` weights matrix
            random_state : int or object
                Numpy random state
            **kwargs
                Keyword arguments for the algorithms

            References
            ----------
                [1] Kenneth E Train. „Discrete choice methods with simulation“. In: Cambridge university press, 2009. Chap Mixed Logit, pp. 153–172.

                [2] Kenneth Train. Qualitative choice analysis. Cambridge, MA: MIT Press, 1986

                [3] Daniel McFadden and Kenneth Train. „Mixed MNL models for discrete response“. In: Journal of applied Econometrics 15.5 (2000), pp. 447–470
        """
        self.logger = logging.getLogger(MixedLogitModel.__name__)
        self.n_object_features = n_object_features
        self.loss_function = likelihood_dict.get(loss_function, None)
        if regularization in ['l1', 'l2']:
            self.regularization = regularization
        else:
            self.regularization = 'l2'
        self._config = None
        self.n_mixtures = n_mixtures
        self.model = None
        self.trace = None
        self.trace_vi = None
        self.Xt = None
        self.Yt = None
        self.p = None

    @property
    def model_configuration(self):
        """
            Constructs the dictionary containing the priors for the weight vectors for the model according to the
            regularization function. The parameters are:
                * **weights** : Distribution of the weigh vectors to evaluates the utility of the objects

            For ``l1`` regularization the priors are:

            .. math::

                \\text{mu}_w \sim \\text{Normal}(\\text{mu}=0, \\text{sd}=5.0) \\\\
                \\text{b}_w \sim \\text{HalfCauchy}(\\beta=1.0) \\\\
                \\text{weights} \sim \\text{Laplace}(\\text{mu}=\\text{mu}_w, \\text{b}=\\text{b}_w)

            For ``l2`` regularization the priors are:

            .. math::

                \\text{mu}_w \sim \\text{Normal}(\\text{mu}=0, \\text{sd}=5.0) \\\\
                \\text{sd}_w \sim \\text{HalfCauchy}(\\beta=1.0) \\\\
                \\text{weights} \sim \\text{Normal}(\\text{mu}=\\text{mu}_w, \\text{sd}=\\text{sd}_w)
        """
        if self._config is None:
            if self.regularization == 'l2':
                weight = pm.Normal
                prior = 'sd'
            elif self.regularization == 'l1':
                weight = pm.Laplace
                prior = 'b'
            self._config = {
                'weights': [weight, {'mu': (pm.Normal, {'mu': 0, 'sd': 5}), prior: (pm.HalfCauchy, {'beta': 1})}]}
            self.logger.info('Creating model with config {}'.format(print_dictionary(self._config)))
        return self._config

    def construct_model(self, X, Y):
        """
            Constructs the mixed logit model by applying priors on weight vectors **weights** as per
            :meth:`model_configuration`. The probability of choosing the object :math:`x_i` from the query set
            :math:`Q = \{x_1, \ldots ,x_n\}` assuming we draw :math:`R` samples of the weight vectors is:

            .. math::

                P(x_i \\lvert Q) = \\frac{1}{R} \sum_{r=1}^R \\frac{exp(U_r(x_i))}{\sum_{x_j \in Q} exp(U_r(x_j))}

            Parameters
            ----------
            X : numpy array
                (n_instances, n_objects, n_features)
                Feature vectors of the objects
            Y : numpy array
                (n_instances, n_objects)
                Preferences in the form of discrete choices for given objects

            Returns
            -------
             model : pymc3 Model :class:`pm.Model`
        """
        with pm.Model() as self.model:
            self.Xt = theano.shared(X)
            self.Yt = theano.shared(Y)
            shapes = {'weights': (self.n_object_features, self.n_mixtures)}
            weights_dict = create_weight_dictionary(self.model_configuration, shapes)
            utility = tt.dot(self.Xt, weights_dict['weights'])
            self.p = tt.mean(ttu.softmax(utility, axis=1), axis=2)
            yl = LogLikelihood('yl', loss_func=self.loss_function, p=self.p, observed=self.Yt)
        self.logger.info("Model construction completed")

    def fit(self, X, Y, sampler='variational', tune=500, draws=500,
            vi_params={"n": 20000, "method": "advi", "callbacks": [CheckParametersConvergence()]}, **kwargs):
        """
            Fit a mixed logit model on the provided set of queries X and choices Y of those objects. The provided
            queries and corresponding preferences are of a fixed size (numpy arrays). For learning this network
            the categorical cross entropy loss function for each object :math:`x_i \in Q` is defined as:

            .. math::

                C_{i} =  -y(i)\log(P_i) \enspace,

            where :math:`y` is ground-truth discrete choice vector of the objects in the given query set :math:`Q`.
            The value :math:`y(i) = 1` if object :math:`x_i` is chosen else :math:`y(i) = 0`.

            Parameters
            ----------
            X : numpy array (n_instances, n_objects, n_features)
                Feature vectors of the objects
            Y : numpy array (n_instances, n_objects)
                Choices for given objects in the query
            sampler : {‘variational’, ‘metropolis’, ‘nuts’}, string
                The sampler used to estimate the posterior mean and mass matrix from the trace

                    * **variational** : Run inference methods to estimate posterior mean and diagonal mass matrix
                    * **metropolis** : Use the MAP as starting point and Metropolis-Hastings sampler
                    * **nuts** : Use the No-U-Turn sampler
            vi_params : dict
                The parameters for the **variational** inference method
            draws : int
                The number of samples to draw. Defaults to 500. The number of tuned samples are discarded by default
            tune : int
                Number of iterations to tune, defaults to 500. Ignored when using 'SMC'. Samplers adjust
                the step sizes, scalings or similar during tuning. Tuning samples will be drawn in addition
                to the number specified in the `draws` argument, and will be discarded unless
                `discard_tuned_samples` is set to False.
            **kwargs :
                Keyword arguments for the fit function of :meth:`pymc3.fit`or :meth:`pymc3.sample`
        """
        self.construct_model(X, Y)
        fit_pymc3_model(self, sampler, draws, tune, vi_params, **kwargs)

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

    def set_tunable_parameters(self, n_mixtures=4, loss_function='', regularization="l1", **point):
        """
            Set tunable parameters of the Mixed Logit model to the values provided.

            Parameters
            ----------
            n_mixtures: int (range : [2, inf])
                The number of logit models (:math:`R`) which are used to estimate the choice probability
            loss_function : string , {‘categorical_crossentropy’, ‘binary_crossentropy’, ’categorical_hinge’}
                Loss function to be used for the discrete choice decision from the query set
            regularization : string, {‘l1’, ‘l2’}, string
               Regularizer function (L1 or L2) applied to the `kernel` weights matrix
            point: dict
                Dictionary containing parameter values which are not tuned for the network
        """
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
        self._config = None
        if len(point) > 0:
            self.logger.warning('This ranking algorithm does not support'
                                ' tunable parameters called: {}'.format(print_dictionary(point)))
