import logging

from csrank.choicefunction.util import generate_complete_pairwise_dataset
from csrank.core.pairwise_svm import PairwiseSVM
from csrank.discretechoice.discrete_choice import DiscreteObjectChooser

logger = logging.getLogger(__name__)


class PairwiseSVMDiscreteChoiceFunction(DiscreteObjectChooser, PairwiseSVM):
    def __init__(
        self,
        C=1.0,
        tol=1e-4,
        normalize=True,
        fit_intercept=True,
        random_state=None,
        **kwargs,
    ):
        """
        Create an instance of the :class:`PairwiseSVM` model for learning a discrete choice function.
        It learns a linear deterministic utility function of the form :math:`U(x) = w \\cdot x`, where :math:`w` is
        the weight vector. It is estimated using *pairwise preferences* generated from the discrete choices.
        The discrete choice for the given query set :math:`Q` is defined as:

        .. math::

            ρ(Q)  = \\operatorname{argmax}_{x \\in Q}  \\; U(x)

        Parameters
        ----------
        C : float, optional
            Penalty parameter of the error term
        tol : float, optional
            Optimization tolerance
        normalize : bool, optional
            If True, the data will be normalized before fitting.
        fit_intercept : bool, optional
            If True, the linear model will also fit an intercept.
        random_state : int, RandomState instance or None, optional
            Seed of the pseudorandom generator or a RandomState instance
        **kwargs
            Keyword arguments for the algorithms


        References
        ----------
            [1] Theodoros Evgeniou, Massimiliano Pontil, and Olivier Toubia.„A convex optimization approach to modeling consumer heterogeneity in conjoint estimation“. In: Marketing Science 26.6 (2007), pp. 805–818.

            [2] Sebastián Maldonado, Ricardo Montoya, and Richard Weber. „Advanced conjoint analysis using feature selection via support vector machines“. In: European Journal of Operational Research 241.2 (2015), pp. 564 –574.
        """
        super().__init__(
            C=C,
            tol=tol,
            normalize=normalize,
            fit_intercept=fit_intercept,
            random_state=random_state,
            **kwargs,
        )
        logger.info("Initializing network")

    def _convert_instances_(self, X, Y):
        logger.debug("Creating the Dataset")
        (
            garbage,
            garbage,
            x_train,
            garbage,
            y_single,
        ) = generate_complete_pairwise_dataset(X, Y)
        del garbage
        assert x_train.shape[1] == self.n_object_features_fit_
        logger.debug("Finished the Dataset with instances {}".format(x_train.shape[0]))
        return x_train, y_single

    def fit(self, X, Y, **kwd):
        self._pre_fit()
        _n_instances, self.n_objects_fit_, self.n_object_features_fit_ = X.shape
        return super().fit(X, Y, **kwd)
