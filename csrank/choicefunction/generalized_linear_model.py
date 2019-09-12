import logging

import numpy as np
import pymc3 as pm
import theano
import theano.tensor as tt
from pymc3.variational.callbacks import CheckParametersConvergence
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state

import csrank.theano_util as ttu
from csrank.choicefunction.util import create_weight_dictionary, BinaryCrossEntropyLikelihood
from csrank.discretechoice.likelihoods import fit_pymc3_model
from csrank.learner import Learner
from csrank.util import print_dictionary
from .choice_functions import ChoiceFunctions


class GeneralizedLinearModel(ChoiceFunctions, Learner):
    def __init__(self, n_object_features, regularization='l2', random_state=None, **kwargs):
        """
            Create an instance of the GeneralizedLinearModel model for learning the choice function. This model is
            adapted from the multinomial logit model :class:`csrank.discretechoice.multinomial_logit_model.MultinomialLogitModel`.
            The utility score for each object in query set :math:`Q` is defined as :math:`U(x) = w \cdot x`,
            where :math:`w` is the weight vector. The probability of choosing an object :math:`x_i` is defined by taking
            sigmoid over the utility scores:

            .. math::

                P(x_i \\lvert Q) = \\frac{1}{1+exp(-U(x_i))}

            The choice set is defined as:

            .. math::

                c(Q) = \{ x_i \in Q \lvert \, P(x_i \\lvert Q) > t \}

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

                [2] Kenneth Train. Qualitative choice analysis. Cambridge, MA: MIT Press, 1986
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
        self.threshold = 0.5

    @property
    def model_configuration(self):
        """
            Constructs the dictionary containing the priors for the weight vectors for the model according to the
            regularization function. The parameters are:
                * **weights** : Weights to evaluates the utility of the objects

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
        if self.regularization == 'l2':
            weight = pm.Normal
            prior = 'sd'
        elif self.regularization == 'l1':
            weight = pm.Laplace
            prior = 'b'
        configuration = {
            'weights': [weight, {'mu': (pm.Normal, {'mu': 0, 'sd': 10}), prior: (pm.HalfCauchy, {'beta': 1})}]}
        self.logger.info('Creating default config {}'.format(print_dictionary(configuration)))
        return configuration

    def construct_model(self, X, Y):
        """
            Constructs the linear logit model which evaluated the utility score as :math:`U(x) = w \cdot x`, where
            :math:`w` is the weight vector. The probability of choosing the object :math:`x_i` from the query set
            :math:`Q = \{x_1, \ldots ,x_n\}` is:

            .. math::

                P_i =  P(x_i \\lvert Q) = \\frac{1}{1+exp(-U(x_i))}

            Parameters
            ----------
            X : numpy array
                (n_instances, n_objects, n_features)
                Feature vectors of the objects
            Y : numpy array
                (n_instances, n_objects)
                Preferences in form of Choices for given objects

            Returns
            -------
             model : pymc3 Model :class:`pm.Model`
        """
        self.logger.info('Creating model_args config {}'.format(print_dictionary(self.model_configuration)))
        with pm.Model() as self.model:
            self.Xt = theano.shared(X)
            self.Yt = theano.shared(Y)
            shapes = {'weights': self.n_object_features}
            # shapes = {'weights': (self.n_object_features, 3)}
            weights_dict = create_weight_dictionary(self.model_configuration, shapes)
            intercept = pm.Normal('intercept', mu=0, sd=10)
            utility = tt.dot(self.Xt, weights_dict['weights']) + intercept
            self.p = ttu.sigmoid(utility)
            yl = BinaryCrossEntropyLikelihood('yl', p=self.p, observed=self.Yt)
        self.logger.info("Model construction completed")

    def fit(self, X, Y, sampler='variational', tune=500, draws=500, tune_size=0.1, thin_thresholds=1,
            vi_params={"n": 20000, "method": "advi", "callbacks": [CheckParametersConvergence()]}, **kwargs):
        """
            Fit a generalized logit model on the provided set of queries X and choices Y of those objects. The
            provided queries and corresponding preferences are of a fixed size (numpy arrays). For learning this network
            the binary cross entropy loss function for each object :math:`x_i \in Q` is defined as:

            .. math::

                C_{i} =  -y(i)\log(P_i) - (1 - y(i))\log(1 - P_i) \enspace,

            where :math:`y` is ground-truth choice vector of the objects in the given query set :math:`Q`.
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
            tune_size: float (range : [0,1])
                Percentage of instances to split off to tune the threshold for the choice function
            thin_thresholds: int
                The number of instances of scores to skip while tuning the threshold
            **kwargs :
                Keyword arguments for the fit function
        """
        if tune_size > 0:
            X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=tune_size, random_state=self.random_state)
            try:
                self._fit(X_train, Y_train, sampler=sampler, vi_params=vi_params, **kwargs)
            finally:
                self.logger.info('Fitting utility function finished. Start tuning threshold.')
                self.threshold = self._tune_threshold(X_val, Y_val, thin_thresholds=thin_thresholds)
        else:
            self._fit(X, Y, sampler=sampler, sample_params={"tune": 2, "draws": 2, "chains": 4, "njobs": 8},
                      vi_params={"n": 20000, "method": "advi", "callbacks": [
                          pm.callbacks.CheckParametersConvergence(diff="absolute", tolerance=0.01, every=50)],
                                 "draws": 500}, **kwargs)
            self.threshold = 0.5

    def _fit(self, X, Y, sampler='variational', tune=500, draws=500,
             vi_params={"n": 20000, "method": "advi", "callbacks": [CheckParametersConvergence()]}, **kwargs):
        self.construct_model(X, Y)
        fit_pymc3_model(self, sampler, draws, tune, vi_params, **kwargs)

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
        """
            Set tunable parameters of the Generalized Linear model to the values provided.

            Parameters
            ----------
            regularization : {‘l1’, ‘l2’}, string
               Regularizer function (L1 or L2) applied to the `kernel` weights matrix
            point: dict
                Dictionary containing parameter values which are not tuned for the network
        """
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
