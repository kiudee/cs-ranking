import logging

import numpy as np
import pymc3 as pm
import theano
import theano.tensor as tt
from pymc3.variational.callbacks import CheckParametersConvergence
from sklearn.cluster import MiniBatchKMeans
from sklearn.utils import check_random_state

import csrank.numpy_util as npu
import csrank.theano_util as ttu
from csrank.discretechoice.likelihoods import create_weight_dictionary, fit_pymc3_model
from csrank.learner import Learner
from csrank.util import print_dictionary
from .discrete_choice import DiscreteObjectChooser
from .likelihoods import likelihood_dict, LogLikelihood


class NestedLogitModel(DiscreteObjectChooser, Learner):
    def __init__(self, n_object_features, n_objects, n_nests=None, loss_function='', regularization='l1', alpha=1e-2,
                 random_state=None, **kwd):
        """
            Create an instance of the Nested Logit model for learning the discrete choice function. This model divides
            objects into disjoint subsets called nests,such that the objects which are similar to each other are in same
            nest. This model structure is 1-layer of hierarchy and the :math:`\lambda` for each nest :math:`B_k` signifies
            the degree of independence and  :math:`1-\lambda` signifies the correlations between the object in it. We
            learn two weight vectors and the  :math:`\lambda s`.

            The probability of choosing an object :math:`x_i` from the given query set :math:`Q` is defined by product
            of choosing the nest in which :math:`x_i` exists and then choosing the the object from the nest.

            .. math::

                P(x_i \\lvert Q) = P_i = P_{i \lvert B_k} P_{B_k} \enspace ,


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

                [3] Kenneth Train and Daniel McFadden. „The goods/leisure tradeoff and disaggregate work trip mode choice models“. In: Transportation research 12.5 (1978), pp. 349–353
        """
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
        self._config = None
        self.cluster_model = None
        self.features_nests = None
        self.model = None
        self.trace = None
        self.trace_vi = None
        self.Xt = None
        self.Yt = None
        self.p = None
        self.y_nests = None
        self.threshold = 5e6

    @property
    def model_configuration(self):
        """
            Constructs the dictionary containing the priors for the weight vectors for the model according to the
            regularization function. The parameters are:
                * **weights** : Weights to evaluates the utility of the objects
                * **weights_k** : Weights to evaluates the utility of the nests

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
                'weights': [weight, {'mu': (pm.Normal, {'mu': 0, 'sd': 5}), prior: (pm.HalfCauchy, {'beta': 1})}],
                'weights_k': [weight, {'mu': (pm.Normal, {'mu': 0, 'sd': 5}), prior: (pm.HalfCauchy, {'beta': 1})}]}
            self.logger.info('Creating model with config {}'.format(print_dictionary(self._config)))
        return self._config

    def create_nests(self, X):
        """
            For allocating the objects to different nests we use the clustering algorithm with number of clusters
            :math:`k` and allocate the similar objects in query set :math:`Q`.

            Parameters
            ----------
            X : numpy array
                (n_instances, n_objects, n_features)
                Feature vectors of the objects in the query sets

            Returns
            -------
            Yn : numpy array
                (n_instances, n_objects) Values for each object implying the nest it belongs to. For example for :math:`2` nests the value 0 implies that object is allocated to nest 1 and value 1 implies it is allocated to nest 2.

        """
        n, n_obj, n_dim = X.shape
        objects = X.reshape(n * n_obj, n_dim)
        if self.cluster_model is None:
            self.cluster_model = MiniBatchKMeans(n_clusters=self.n_nests, random_state=self.random_state).fit(objects)
            self.features_nests = self.cluster_model.cluster_centers_
            prediction = self.cluster_model.labels_
        else:
            prediction = self.cluster_model.predict(objects)
        Yn = []
        for i in np.arange(0, n * n_obj, step=n_obj):
            nest_ids = prediction[i:i + n_obj]
            Yn.append(nest_ids)
        Yn = np.array(Yn)
        return Yn

    def _eval_utility(self, weights):
        utility = tt.zeros(tuple(self.y_nests.shape))
        for i in range(self.n_nests):
            rows, cols = tt.eq(self.y_nests, i).nonzero()
            utility = tt.set_subtensor(utility[rows, cols], tt.dot(self.Xt[rows, cols], weights[i]))
        return utility

    def get_probabilities(self, utility, lambda_k, utility_k):
        """
            This method calculates the probability of choosing an object from the query set using the following parameters of the model which are used:

                * **weights** (:math:`w`): Weights to get the utility of the object :math:`Y_i = U(x_i) = w \cdot x_i`
                * **weights_k** (:math:`w_k`): Weights to get the utility of the next  :math:`W_k = U_k(x) = w_k \cdot c_k`, where :math:`c_k` is the center of the object space of nest :math:`B_k`
                * **lambda_k** (:math:`\lambda_k`): Lambda is the measure of independence amongst the obejcts in the nest :math:`B_k`

            The probability of choosing the object  :math:`x_i` from the query set :math:`Q`:

            .. math::
                P_i = \\frac{\\boldsymbol{e}^{ ^{Y_i} /_{\lambda_k}}}{\sum_{j \in B_k} \\boldsymbol{e}^{^{Y_j} /_{\lambda_k}}} \\frac {\\boldsymbol{e}^{W_k + \lambda_k I_k}} {\sum_{\\ell = 1}^{K} \\boldsymbol{e}^{ W_{\\ell } + \lambda_{\\ell} I_{\\ell}}} \quad i \in B_k  \enspace , \\\\
                where,\enspace I_k = \ln \sum_{ j \in B_k} \\boldsymbol{e}^{^{Y_j} /_{\lambda_k}}


            Parameters
            ----------
            utility : theano tensor
                (n_instances, n_objects)
                Utility :math:`Y_i` of the objects :math:`x_i \in Q` in the query sets
            lambda_k : theano tensor (range : [alpha, 1.0])
                (n_nests)
                Measure of independence amongst the obejcts in each nests
            utility_k : theano tensor
                (n_instances, n_nests)
                Utilities of the nests :math:`B_k \in \mathcal{B}`

            Returns
            -------
            p : theano tensor
                (n_instances, n_objects)
                Choice probabilities :math:`P_i` of the objects :math:`x_i \in Q` in the query sets

        """
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

    def _eval_utility_np(self, x_t, y_nests, weights):
        utility = np.zeros(tuple(y_nests.shape))
        for i in range(self.n_nests):
            rows, cols = np.where(y_nests == i)
            utility[rows, cols] = np.dot(x_t[rows, cols], weights[i])
        return utility

    def _get_probabilities_np(self, Y_n, utility, lambda_k, utility_k):
        n_instances, n_objects = Y_n.shape
        pni_k = np.zeros((n_instances, n_objects))
        ivm = np.zeros((n_instances, self.n_nests))
        for i in range(self.n_nests):
            sub_tensor = np.copy(utility)
            sub_tensor[np.where(Y_n != i)] = -1e50
            ink = npu.logsumexp(sub_tensor)
            pni_k[np.where(Y_n == i)] = np.exp(sub_tensor - ink)[np.where(Y_n == i)]
            ivm[:, i] = lambda_k[i] * ink[:, 0] + utility_k[i]
        pk = np.exp(ivm - npu.logsumexp(ivm))
        pn_k = np.zeros((n_instances, n_objects))
        for i in range(self.n_nests):
            rows, cols = np.where(Y_n == i)
            p = np.ones((n_instances, n_objects)) * pk[:, i][:, None]
            pn_k[rows, cols] = p[rows, cols]
        p = pni_k * pn_k
        return p

    def construct_model(self, X, Y):
        """
            Constructs the nested logit model by applying priors on weight vectors **weights** and **weights_k** as per
            :meth:`model_configuration`. Then we apply a uniform prior to the :math:`\lambda s`, i.e.
            :math:`\lambda s \sim Uniform(\\text{alpha}, 1.0)`.The probability of choosing the object :math:`x_i` from
            the query set :math:`Q = \{x_1, \ldots ,x_n\}` is evaluated in :meth:`get_probabilities`.

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
        if np.prod(X.shape) > self.threshold:
            l = int(self.threshold / np.prod(X.shape[1:]))
            indices = self.random_state.choice(X.shape[0], l, replace=False)
            X = X[indices, :, :]
            Y = Y[indices, :]
        self.logger.info("Train Set instances {} objects {} features {}".format(*X.shape))
        y_nests = self.create_nests(X)
        with pm.Model() as self.model:
            self.Xt = theano.shared(X)
            self.Yt = theano.shared(Y)
            self.y_nests = theano.shared(y_nests)
            shapes = {'weights': self.n_object_features, 'weights_k': self.n_object_features}

            weights_dict = create_weight_dictionary(self.model_configuration, shapes)
            lambda_k = pm.Uniform('lambda_k', self.alpha, 1.0, shape=self.n_nests)
            weights = (weights_dict['weights'] / lambda_k[:, None])
            utility = self._eval_utility(weights)
            utility_k = tt.dot(self.features_nests, weights_dict['weights_k'])
            self.p = self.get_probabilities(utility, lambda_k, utility_k)

            yl = LogLikelihood('yl', loss_func=self.loss_function, p=self.p, observed=self.Yt)
        self.logger.info("Model construction completed")

    def fit(self, X, Y, sampler='variational', tune=500, draws=500,
            vi_params={"n": 20000, "method": "advi", "callbacks": [CheckParametersConvergence()]}, **kwargs):
        """
            Fit a nested logit model on the provided set of queries X and choices Y of those objects. The provided
            queries and corresponding preferences are of a fixed size (numpy arrays). For learning this network the
            categorical cross entropy loss function for each object :math:`x_i \in Q` is defined as:

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
        y_nests = self.create_nests(X)
        mean_trace = dict(pm.summary(self.trace)['mean'])
        weights = np.array([mean_trace['weights__{}'.format(i)] for i in range(self.n_object_features)])
        weights_k = np.array([mean_trace['weights_k__{}'.format(i)] for i in range(self.n_object_features)])
        lambda_k = np.array([mean_trace['lambda_k__{}'.format(i)] for i in range(self.n_nests)])
        weights = (weights / lambda_k[:, None])
        utility_k = np.dot(self.features_nests, weights_k)
        utility = self._eval_utility_np(X, y_nests, weights)
        scores = self._get_probabilities_np(y_nests, utility, lambda_k, utility_k)
        return scores

    def predict(self, X, **kwargs):
        return super().predict(X, **kwargs)

    def predict_scores(self, X, **kwargs):
        return super().predict_scores(X, **kwargs)

    def predict_for_scores(self, scores, **kwargs):
        return DiscreteObjectChooser.predict_for_scores(self, scores, **kwargs)

    def set_tunable_parameters(self, alpha=None, n_nests=None, loss_function='', regularization="l1", **point):
        """
            Set tunable parameters of the Nested Logit model to the values provided.

            Parameters
            ----------
            alpha: float (range : [0,1])
                The lower bound of the correlations between the objects in a nest
            n_nests: int (range : [2,n_objects])
                The number of nests in which the objects are divided
            loss_function : string , {‘categorical_crossentropy’, ‘binary_crossentropy’, ’categorical_hinge’}
                Loss function to be used for the discrete choice decision from the query set
            regularization : string, {‘l1’, ‘l2’}, string
               Regularizer function (L1 or L2) applied to the `kernel` weights matrix
            point: dict
                Dictionary containing parameter values which are not tuned for the network
        """
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
        self._config = None
        if len(point) > 0:
            self.logger.warning('This ranking algorithm does not support tunable parameters'
                                ' called: {}'.format(print_dictionary(point)))
