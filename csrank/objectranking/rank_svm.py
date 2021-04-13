import logging

from csrank.core.pairwise_svm import PairwiseSVM
from csrank.objectranking.object_ranker import ObjectRanker
from ..dataset_reader.objectranking.util import generate_complete_pairwise_dataset

__all__ = ["RankSVM"]
logger = logging.getLogger(__name__)


class RankSVM(ObjectRanker, PairwiseSVM):
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
        Create an instance of the :class:`PairwiseSVM` model for learning a object ranking function.
        It learns a linear deterministic utility function of the form :math:`U(x) = w \\cdot x`, where :math:`w` is
        the weight vector. It is estimated using *pairwise preferences* generated from the rankings.
        The ranking for the given query set :math:`Q` is defined as:

        .. math::

            ρ(Q)  = \\operatorname{argsort}_{x \\in Q}  \\; U(x)

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
            [1] Joachims, T. (2002, July). "Optimizing search engines using clickthrough data.", Proceedings of the eighth ACM SIGKDD international conference on Knowledge discovery and data mining (pp. 133-142). ACM.
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

    def fit(self, X, Y, **kwargs):
        self._pre_fit()
        _n_instances, self.n_objects_fit_, self.n_object_features_fit_ = X.shape
        return super().fit(X, Y, **kwargs)

    def _convert_instances_(self, X, Y):
        logger.debug("Creating the Dataset")
        (
            x_train,
            garbage,
            garbage,
            garbage,
            y_single,
        ) = generate_complete_pairwise_dataset(X, Y)
        del garbage
        assert x_train.shape[1] == self.n_object_features_fit_
        logger.debug("Finished the Dataset with instances {}".format(x_train.shape[0]))
        return x_train, y_single
