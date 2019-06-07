import logging
from itertools import combinations

import numpy as np
import pymc3 as pm
import theano
import theano.tensor as tt
from pymc3.variational.callbacks import CheckParametersConvergence
from sklearn.utils import check_random_state

import csrank.numpy_util as npu
import csrank.theano_util as ttu
from csrank.learner import Learner
from csrank.util import print_dictionary
from .discrete_choice import DiscreteObjectChooser
from .likelihoods import likelihood_dict, LogLikelihood, create_weight_dictionary, fit_pymc3_model


class PairedCombinatorialLogit(DiscreteObjectChooser, Learner):

    def __init__(self, n_object_features, n_objects, loss_function='', regularization='l2', alpha=5e-2,
                 random_state=None, **kwd):
        """
            Create an instance of the Paired Combinatorial Logit model for learning the discrete choice function. This
            model considering each pair of objects as a different nest allowing unique covariances for each pair of objects,
            and each object is a member of :math:`n - 1` nests. This model structure is 1-layer of hierarchy and the
            :math:`\lambda` for each nest :math:`B_k` signifies the degree of independence and  :math:`1-\lambda` signifies
            the correlations between the object in it. We learn two weight vectors and the  :math:`\lambda s`.
                * **weights** (:math:`w`): Weights to get the utility of the object :math:`Y_i = U(x_i) = w \cdot x_i`
                * **lambda_k** (:math:`\lambda_k`): Lambda for nest nest :math:`B_k` for correlations between the obejcts.

            The probability of choosing an object :math:`x_i` from the given query set :math:`Q` is defined by product
            of choosing the nest in which :math:`x_i` exists and then choosing the the object from the nest.

            .. math::

                P(x_i \\lvert Q) = P_i = \sum_{\substack{B_k \in \mathcal{B} \\ i \in B_k}}P_{i \\lvert B_k} P_{B_k} \enspace ,


            The discrete choice for the given query set :math:`Q` is defined as:

            .. math::

                dc(Q) := \operatorname{argmax}_{x_i \in Q }  \; P(x_i \\lvert Q)

            Parameters
            ----------
            n_object_features : int
                Number of features of the object space
            n_objects: int
                Number of objects in each query set
            n_nests : int range : [2,n_objects/2]
                The number of nests/subsets in which the objects are divided
            loss_function : string , {‘categorical_crossentropy’, ‘binary_crossentropy’, ’categorical_hinge’}
                Loss function to be used for the discrete choice decision from the query set
            regularization : string, {‘l1’, ‘l2’}, string
               Regularizer function (L1 or L2) applied to the `kernel` weights matrix
            alpha: float (range : [0,1])
                The lower bound of the correlations between the objects in a nest
            random_state : int or object
                Numpy random state
            **kwargs
                Keyword arguments for the algorithms

            References
            ----------
                [1] Kenneth E Train. „Discrete choice methods with simulation“. In: Cambridge university press, 2009. Chap GEV, pp. 87–111.

                [2] Kenneth Train. Qualitative choice analysis. Cambridge, MA: MIT Press, 1986

                [3] Chaushie Chu. „A paired combinatorial logit model for travel demand analysis“. In: Proceedings of the fifth world conference on transportation research. Vol. 4.1989, pp. 295–309
        """
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
        self._config = None
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

            Returns
            -------
                configuration : dict
                    Dictionary containing the priors applies on the weights
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

    #

    def get_probabilities(self, utility, lambda_k):
        """
            This method calculates the probability of choosing an object from the query set using the following parameters of the model which are used:

                * **weights** (:math:`w`): Weights to get the utility of the object :math:`Y_i = U(x_i) = w \cdot x_i`
                * **lambda_k** (:math:`\lambda_k`): Lambda is the measure of independence amongst the obejcts in the nest :math:`B_k`

            The probability of choosing the object  :math:`x_i` from the query set :math:`Q`:

            .. math::
                    P_i = \sum_{j \in I \setminus i} P_{{i} \\lvert {ij}} P_{ij} \enspace where, \\\\
                    P_{i \\lvert ij} = \\frac{\\boldsymbol{e}^{^{Y_i} /_{\lambda_{ij}}}}{\\boldsymbol{e}^{^{Y_i} /_{\lambda_{ij}}} + \\boldsymbol{e}^{^{Y_j} /_{\lambda_{ij}}}} \enspace ,\\\\
                    P_{ij} = \\frac{{\\left( \\boldsymbol{e}^{^{V_i}/{\lambda_{ij}}} + \\boldsymbol{e}^{^{V_j}/{\lambda_{ij}}}  \\right)}^{\lambda_{ij}}}{\sum_{k=1}^{n-1} \sum_{\ell = k + 1}^{n} {\\left( \\boldsymbol{e}^{^{V_k}/{\lambda_{k\ell}}} + \\boldsymbol{e}^{^{V_{\ell}}/{\lambda_{k\ell}}}  \\right)}^{\lambda_{k\ell}}}


            Parameters
            ----------
            utility : theano tensor
                (n_instances, n_objects)
                Utility :math:`Y_i` of the objects :math:`x_i \in Q` in the query sets
            lambda_k : theano tensor (range : [alpha, 1.0])
                (n_nests)
                Measure of independence amongst the obejcts in each nests

            Returns
            -------
            p : theano tensor
                (n_instances, n_objects)
                Choice probabilities :math:`P_i` of the objects :math:`x_i \in Q` in the query sets

        """
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

    def _get_probabilities_np(self, utility, lambda_k):
        n_objects = self.n_objects
        nests_indices = self.nests_indices
        n_nests = self.n_nests
        temp_lambdas = np.ones((n_objects, n_objects), lambda_k.dtype)
        temp_lambdas[nests_indices[:, 0], nests_indices[:, 1]] = temp_lambdas.T[
            nests_indices[:, 0], nests_indices[:, 1]] = lambda_k
        uti_per_nest = np.transpose((utility[:, None] / temp_lambdas), (0, 2, 1))
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
        """
            Constructs the nested logit model by applying priors on weight vectors **weights** as per :meth:`model_configuration`.
            Then we apply a uniform prior to the :math:`\lambda s`, i.e. :math:`\lambda s \sim Uniform(\\text{alpha}, 1.0)`.
            The probability of choosing the object :math:`x_i` from the query set :math:`Q = \{x_1, \ldots ,x_n\}` is
            evaluated in :meth:`get_probabilities`.

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
            shapes = {'weights': self.n_object_features}
            weights_dict = create_weight_dictionary(self.model_configuration, shapes)
            lambda_k = pm.Uniform('lambda_k', self.alpha, 1.0, shape=self.n_nests)
            utility = tt.dot(self.Xt, weights_dict['weights'])
            self.p = self.get_probabilities(utility, lambda_k)
            yl = LogLikelihood('yl', loss_func=self.loss_function, p=self.p, observed=self.Yt)
        self.logger.info("Model construction completed")

    def fit(self, X, Y, sampler='variational', tune=500, draws=500,
            vi_params={"n": 20000, "method": "advi", "callbacks": [CheckParametersConvergence()]}, **kwargs):
        """
           Fit a paired combinatorial logit  model on the provided set of queries X and choices Y of those objects. The
           provided queries and corresponding preferences are of a fixed size (numpy arrays). For learning this network
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
        mean_trace = dict(pm.summary(self.trace)['mean'])
        weights = np.array([mean_trace['weights__{}'.format(i)] for i in range(self.n_object_features)])
        lambda_k = np.array([mean_trace['lambda_k__{}'.format(i)] for i in range(self.n_nests)])
        utility = np.dot(X, weights)
        p = self._get_probabilities_np(utility, lambda_k)
        return p

    def predict(self, X, **kwargs):
        return super().predict(X, **kwargs)

    def predict_scores(self, X, **kwargs):
        return super().predict_scores(X, **kwargs)

    def predict_for_scores(self, scores, **kwargs):
        return DiscreteObjectChooser.predict_for_scores(self, scores, **kwargs)

    def set_tunable_parameters(self, alpha=5e-2, loss_function='', regularization='l2', **point):
        """
            Set tunable parameters of the Paired Combinatorial logit model to the values provided.

            Parameters
            ----------
            alpha: float (range : [0,1])
                The lower bound of the correlations between the objects in a nest
            loss_function : string , {‘categorical_crossentropy’, ‘binary_crossentropy’, ’categorical_hinge’}
                Loss function to be used for the discrete choice decision from the query set
            regularization : string, {‘l1’, ‘l2’}, string
               Regularizer function (L1 or L2) applied to the `kernel` weights matrix
            point: dict
                Dictionary containing parameter values which are not tuned for the network
        """
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
        self._config = None
        if len(point) > 0:
            self.logger.warning('This ranking algorithm does not support tunable parameters'
                                ' called: {}'.format(print_dictionary(point)))
