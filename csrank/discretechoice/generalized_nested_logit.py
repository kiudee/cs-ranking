from itertools import product
import logging

import numpy as np
from sklearn.utils import check_random_state

from csrank.learner import Learner
import csrank.numpy_util as npu
import csrank.theano_util as ttu
from csrank.util import print_dictionary
from .discrete_choice import DiscreteObjectChooser
from .likelihoods import create_weight_dictionary
from .likelihoods import fit_pymc3_model
from .likelihoods import likelihood_dict
from .likelihoods import LogLikelihood

try:
    import pymc3 as pm
    from pymc3.variational.callbacks import CheckParametersConvergence
except ImportError:
    from csrank.util import MissingExtraError

    raise MissingExtraError("pymc3", "probabilistic")

try:
    import theano
    from theano import tensor as tt
except ImportError:
    from csrank.util import MissingExtraError

    raise MissingExtraError("theano", "probabilistic")

logger = logging.getLogger(__name__)


class GeneralizedNestedLogitModel(DiscreteObjectChooser, Learner):
    def __init__(
        self,
        n_nests=None,
        loss_function="None",
        regularization="l2",
        alpha=5e-2,
        random_state=None,
        **kwd,
    ):
        """
        Create an instance of the Generalized Nested Logit model for learning the discrete choice function. This
        model divides objects into subsets called nests, such that the each object is associtated to each nest to some degree.
        This model structure is 1-layer of hierarchy and the :math:`\\lambda` for each nest :math:`B_k` signifies the degree of independence
        and  :math:`1-\\lambda` signifies the correlations between the object in it. We learn two weight vectors and the  :math:`\\lambda s`.
        The probability of choosing an object :math:`x_i` from the given query set :math:`Q` is defined by product
        of choosing the nest in which :math:`x_i` exists and then choosing the the object from the nest.

        .. math::

            P(x_i \\lvert Q) = P_i = \\sum_{\\substack{B_k \\in \\mathcal{B} \\ i \\in B_k}}P_{i \\lvert B_k} P_{B_k} \\enspace ,


        The discrete choice for the given query set :math:`Q` is defined as:

        .. math::

            dc(Q) := \\operatorname{argmax}_{x_i \\in Q }  \\; P(x_i \\lvert Q)

        Parameters
        ----------
        n_objects: int
            Number of objects in each query set
        n_nests : int range : [2,n_objects/2]
            The number of nests/subsets in which the objects are divided.
            This may not surpass half the amount of objects this model will
            be trained on.
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

            [3] Chieh-Hua Wen and Frank S Koppelman. „The generalized nested logit model“. In: Transportation Research Part B: Methodological 35.7 (2001), pp. 627–641

        """

        self.n_nests = n_nests
        self.alpha = alpha
        self.loss_function = loss_function

        self.random_state = random_state
        known_regularization_functions = {"l1", "l2"}
        if regularization not in known_regularization_functions:
            raise ValueError(
                f"Regularization function {regularization} is unknown. Must be one of {known_regularization_functions}"
            )
        self.regularization = regularization

    @property
    def model_configuration(self):
        """
            Constructs the dictionary containing the priors for the weight vectors for the model according to the
            regularization function. The parameters are:
                * **weights** : Weights to evaluates the utility of the objects
                * **weights_k** : Weights to evaluates the fractional allocation of each object in :math:'Q' to each nest

            For ``l1`` regularization the priors are:

            .. math::

                \\text{mu}_w \\sim \\text{Normal}(\\text{mu}=0, \\text{sd}=5.0) \\\\
                \\text{b}_w \\sim \\text{HalfCauchy}(\\beta=1.0) \\\\
                \\text{weights} \\sim \\text{Laplace}(\\text{mu}=\\text{mu}_w, \\text{b}=\\text{b}_w)

            For ``l2`` regularization the priors are:

            .. math::

                \\text{mu}_w \\sim \\text{Normal}(\\text{mu}=0, \\text{sd}=5.0) \\\\
                \\text{sd}_w \\sim \\text{HalfCauchy}(\\beta=1.0) \\\\
                \\text{weights} \\sim \\text{Normal}(\\text{mu}=\\text{mu}_w, \\text{sd}=\\text{sd}_w)

            Returns
            -------
                configuration : dict
                    Dictionary containing the priors applies on the weights
        """
        if not hasattr(self, "config_"):
            if self.regularization == "l2":
                weight = pm.Normal
                prior = "sd"
            elif self.regularization == "l1":
                weight = pm.Laplace
                prior = "b"
            self.config_ = {
                "weights": [
                    weight,
                    {
                        "mu": (pm.Normal, {"mu": 0, "sd": 5}),
                        prior: (pm.HalfCauchy, {"beta": 1}),
                    },
                ],
                "weights_ik": [
                    weight,
                    {
                        "mu": (pm.Normal, {"mu": 0, "sd": 5}),
                        prior: (pm.HalfCauchy, {"beta": 1}),
                    },
                ],
            }
            logger.info(
                "Creating model with config {}".format(print_dictionary(self.config_))
            )
        return self.config_

    def get_probabilities(self, utility, lambda_k, alpha_ik):
        """
            This method calculates the probability of choosing an object from the query set using the following parameters of the model which are used:

                * **weights** (:math:`w`): Weights to get the utility of the object :math:`Y_i = U(x_i) = w \\cdot x_i`
                * **weights_k** (:math:`w_k`): Weights to get fractional allocation of each object :math:'x_j'  in :math:'Q' to each nest math:`B_k` as :math:`\\alpha_{ik} = w_k \\cdot x_i`.
                * **lambda_k** (:math:`\\lambda_k`): Lambda for nest :math:`B_k` for correlations between the obejcts.

            The probability of choosing the object :math:`x_i` from the query set :math:`Q`:

            .. math::
                P_i = \\sum_{\\substack{B_k \\in \\mathcal{B} \\ i \\in B_k}} P_{i \\lvert {B_k}} P_{B_k} \\enspace where, \\\\
                P_{B_k} = \\frac{{\\left(\\sum_{j \\in B_k} {\\left(\\alpha_{jk} \\boldsymbol{e}^{V_j} \\right)}^ {^{1}/{\\lambda_k}} \\right)}^{\\lambda_k}}{\\sum_{\\ell = 1}^{K} {\\left( \\sum_{j \\in B_{\\ell}} {\\left( \\alpha_{j\\ell} \\boldsymbol{e}^{V_j} \\right)}^{^{1}/{\\lambda_\\ell}} \\right)^{\\lambda_{\\ell}}}} \\\\
                P_{{i} \\lvert {B_k}} = \\frac{{\\left(\\alpha_{ik} \\boldsymbol{e}^{V_i} \\right)}^{^{1}/{\\lambda_k}}}{\\sum_{j \\in B_k} {\\left(\\alpha_{jk} \\boldsymbol{e}^{V_j} \\right)}^{^{1}/{\\lambda_k}}} \\enspace ,


            Parameters
            ----------
            utility : theano tensor
                (n_instances, n_objects)
                Utility :math:`Y_i` of the objects :math:`x_i \\in Q` in the query sets
            lambda_k : theano tensor (range : [alpha, 1.0])
                (n_nests)
                Measure of independence amongst the obejcts in each nests
            alpha_ik : theano tensor
                (n_instances, n_objects, n_nests)
                Fractional allocation of each object :math:`x_i` in each nest math:`B_k`

            Returns
            -------
            p : theano tensor
                (n_instances, n_objects)
                Choice probabilities :math:`P_i` of the objects :math:`x_i \\in Q` in the query sets

        """
        n_nests = self.n_nests
        n_instances, n_objects = utility.shape
        pik = tt.zeros((n_instances, n_objects, n_nests))
        sum_per_nest = tt.zeros((n_instances, n_nests))
        for i in range(n_nests):
            uti = (utility + tt.log(alpha_ik[:, :, i])) * 1 / lambda_k[i]
            sum_n = ttu.logsumexp(uti)
            pik = tt.set_subtensor(pik[:, :, i], tt.exp(uti - sum_n))
            sum_per_nest = tt.set_subtensor(
                sum_per_nest[:, i], sum_n[:, 0] * lambda_k[i]
            )
        pnk = tt.exp(sum_per_nest - ttu.logsumexp(sum_per_nest))
        pnk = pnk[:, None, :]
        p = pik * pnk
        p = p.sum(axis=2)
        return p

    def _get_probabilities_np(self, utility, lambda_k, alpha_ik):
        n_nests = self.n_nests
        n_instances, n_objects = utility.shape
        pik = np.zeros((n_instances, n_objects, n_nests))
        sum_per_nest_x = np.zeros((n_instances, n_nests))
        for i in range(n_nests):
            uti = (utility + np.log(alpha_ik[:, :, i])) * 1 / lambda_k[i]
            sum_n = npu.logsumexp(uti)
            pik[:, :, i] = np.exp(uti - sum_n)
            sum_per_nest_x[:, i] = sum_n[:, 0] * lambda_k[i]
        pnk = np.exp(sum_per_nest_x - npu.logsumexp(sum_per_nest_x))
        pnk = pnk[:, None, :]
        p = pik * pnk
        p = p.sum(axis=2)
        return p

    def construct_model(self, X, Y):
        """
        Constructs the nested logit model by applying priors on weight vectors **weights** and **weights_k** as per
        :meth:`model_configuration`. Then we apply a uniform prior to the :math:`\\lambda s`, i.e.
        :math:`\\lambda s \\sim Uniform(\\text{alpha}, 1.0)`.The probability of choosing the object :math:`x_i` from the
        query set :math:`Q = \\{x_1, \\ldots ,x_n\\}` is evaluated in :meth:`get_probabilities`.

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
        self.trace_ = None
        self.trace_vi_ = None
        self.random_state_ = check_random_state(self.random_state)
        self.loss_function_ = likelihood_dict.get(self.loss_function, None)
        self.threshold_ = 43e5
        if np.prod(X.shape) > self.threshold_:
            upper_bound = int(self.threshold_ / np.prod(X.shape[1:]))
            indices = self.random_state_.choice(X.shape[0], upper_bound, replace=False)
            X = X[indices, :, :]
            Y = Y[indices, :]
        logger.info("Train Set instances {} objects {} features {}".format(*X.shape))
        with pm.Model() as self.model:
            self.Xt_ = theano.shared(X)
            self.Yt_ = theano.shared(Y)
            shapes = {
                "weights": self.n_object_features_fit_,
                "weights_ik": (self.n_object_features_fit_, self.n_nests),
            }
            weights_dict = create_weight_dictionary(self.model_configuration, shapes)

            alpha_ik = tt.dot(self.Xt_, weights_dict["weights_ik"])
            alpha_ik = ttu.softmax(alpha_ik, axis=2)
            utility = tt.dot(self.Xt_, weights_dict["weights"])
            lambda_k = pm.Uniform("lambda_k", self.alpha, 1.0, shape=self.n_nests)
            self.p_ = self.get_probabilities(utility, lambda_k, alpha_ik)
            LogLikelihood(
                "yl", loss_func=self.loss_function_, p=self.p_, observed=self.Yt_
            )
        logger.info("Model construction completed")

    def fit(
        self,
        X,
        Y,
        sampler="variational",
        tune=500,
        draws=500,
        vi_params={
            "n": 20000,
            "method": "advi",
            "callbacks": [CheckParametersConvergence()],
        },
        **kwargs,
    ):
        """
        Fit a generalized nested logit model on the provided set of queries X and choices Y of those objects. The
        provided queries and corresponding preferences are of a fixed size (numpy arrays). For learning this network
        the categorical cross entropy loss function for each object :math:`x_i \\in Q` is defined as:

        .. math::

            C_{i} =  -y(i)\\log(P_i) \\enspace,

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
        self._pre_fit()
        _n_instances, self.n_objects_fit_, self.n_object_features_fit_ = X.shape
        if self.n_nests is None:
            # TODO this looks like a bug to me, but it was already done this way
            # before (moved out of __init__). The `n_objects` summand probably
            # should be removed.
            self.n_nests = self.n_objects_fit_ + int(self.n_objects_fit_ / 2)
        self.construct_model(X, Y)
        fit_pymc3_model(self, sampler, draws, tune, vi_params, **kwargs)
        return self

    def _predict_scores_fixed(self, X, **kwargs):
        mean_trace = dict(pm.summary(self.trace_)["mean"])
        weights = np.array(
            [
                mean_trace["weights[{}]".format(i)]
                for i in range(self.n_object_features_fit_)
            ]
        )
        lambda_k = np.array(
            [mean_trace["lambda_k[{}]".format(i)] for i in range(self.n_nests)]
        )
        weights_ik = np.zeros((self.n_object_features_fit_, self.n_nests))
        for i, k in product(range(self.n_object_features_fit_), range(self.n_nests)):
            weights_ik[i][k] = mean_trace["weights_ik[{},{}]".format(i, k)]
        alpha_ik = np.dot(X, weights_ik)
        alpha_ik = npu.softmax(alpha_ik, axis=2)
        utility = np.dot(X, weights)
        p = self._get_probabilities_np(utility, lambda_k, alpha_ik)
        return p
