import logging

import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.utils import check_random_state

from csrank.learner import Learner
from csrank.numpy_util import normalize
from csrank.objectranking.object_ranker import ObjectRanker
from ..dataset_reader.objectranking.util import complete_linear_regression_dataset

__all__ = ["ExpectedRankRegression"]

logger = logging.getLogger(__name__)


class ExpectedRankRegression(ObjectRanker, Learner):
    def __init__(
        self,
        alpha=0.0,
        l1_ratio=0.5,
        tol=1e-4,
        normalize=True,
        fit_intercept=True,
        random_state=None,
        **kwargs,
    ):
        """
        Create an expected rank regression model.
        This model normalizes the ranks to [0, 1] and treats them as regression target. For ``α = 0`` we employ
        simple linear regression. For α > 0 the model becomes ridge regression (when ``l1_ratio = 0``) or
        elastic net (when ``l1_ratio > 0``). The target for an object :math:`x_k \\in Q` is:


        .. math::
            r(x_k) = \\frac{\\pi(k)}{n} \\quad ,

        where :math:`\\pi(k)` is the rank of the :math:`x_k`. The regression model learns a function
        :math:`F \\colon \\mathcal{X} \\to \\mathbb{R}`. The ranking for the given query set :math:`Q` defined as:

        .. math::
            ρ(Q)  = \\operatorname{argsort}_{x \\in Q}  \\; -1*F(x)


        Parameters
        ----------
        alpha : float, optional
            Regularization strength
        l1_ratio : float, optional
            Ratio between pure L2 (=0) or pure L1 (=1) regularization.
        tol : float, optional
            Optimization tolerance
        normalize : bool, optional
            If True, the regressors will be normalized before fitting.
        fit_intercept : bool, optional
            If True, the linear model will also fit an intercept.
        random_state : int, RandomState instance or None, optional
            Seed of the pseudorandom generator or a RandomState instance
        **kwargs
            Keyword arguments for the algorithms

        References
        ----------
            [1] Kamishima, T., Kazawa, H., & Akaho, S. (2005, November). "Supervised ordering-an empirical survey.", Fifth IEEE International Conference on Data Mining.
        """
        self.normalize = normalize
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.random_state = random_state

    def _pre_fit(self):
        super()._pre_fit()
        self.random_state_ = check_random_state(self.random_state)

    def fit(self, X, Y, **kwargs):
        """
        Fit an ExpectedRankRegression on the provided set of queries X and preferences Y of those objects.
        The provided queries and corresponding preferences are of a fixed size (numpy arrays).

        Parameters
        ----------
        X : numpy array
            (n_instances, n_objects, n_features)
            Feature vectors of the objects
        Y : numpy array
            (n_instances, n_objects)
            Rankings of the given objects
        **kwargs
            Keyword arguments for the fit function
        """
        self._pre_fit()
        logger.debug("Creating the Dataset")
        x_train, y_train = complete_linear_regression_dataset(X, Y)
        logger.debug("Finished the Dataset")
        if self.alpha < 1e-3:
            self.model_ = LinearRegression(
                normalize=self.normalize, fit_intercept=self.fit_intercept
            )
            logger.info("LinearRegression")
        else:
            if self.l1_ratio >= 0.01:
                self.model_ = ElasticNet(
                    alpha=self.alpha,
                    l1_ratio=self.l1_ratio,
                    normalize=self.normalize,
                    tol=self.tol,
                    fit_intercept=self.fit_intercept,
                    random_state=self.random_state_,
                )
                logger.info("Elastic Net")
            else:
                self.model_ = Ridge(
                    alpha=self.alpha,
                    normalize=self.normalize,
                    tol=self.tol,
                    fit_intercept=self.fit_intercept,
                    random_state=self.random_state_,
                )
                logger.info("Ridge")
        logger.debug("Finished Creating the model, now fitting started")
        self.model_.fit(x_train, y_train)
        self.weights_ = self.model_.coef_.flatten()
        if self.fit_intercept:
            self.weights_ = np.append(self.weights_, self.model_.intercept_)
        logger.debug("Fitting Complete")
        return self

    def _predict_scores_fixed(self, X, **kwargs):
        n_instances, n_objects, n_features = X.shape
        logger.info("For Test instances {} objects {} features {}".format(*X.shape))
        X1 = X.reshape(n_instances * n_objects, n_features)
        scores = n_objects - self.model_.predict(X1)
        scores = scores.reshape(n_instances, n_objects)
        scores = normalize(scores)
        logger.info("Done predicting scores")
        return scores
