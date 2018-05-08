import logging

import numpy as np
from sklearn.linear_model import LinearRegression, ElasticNet, Ridge
from sklearn.utils import check_random_state

from csrank.objectranking.object_ranker import ObjectRanker
from csrank.tunable import Tunable
from csrank.util import normalize, print_dictionary
from ..dataset_reader.objectranking.util import \
    complete_linear_regression_dataset

__all__ = ['ExpectedRankRegression']


class ExpectedRankRegression(ObjectRanker, Tunable):
    _tunable = None

    def __init__(self, n_features, alpha=0.0, l1_ratio=0.5, tol=1e-4,
                 normalize=True, fit_intercept=True, random_state=None,
                 **kwargs):
        """Create an expected rank regression model.

        This model normalizes the ranks to [0, 1] and treats them as regression
        target. For α = 0 we employ simple linear regression. For α > 0 the
        model becomes ridge regression (when l1_ratio = 0) or elastic net
        (when l1_ratio > 0).

        Parameters
        ----------
        n_features : int
            Number of features of the object space
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
        .. [1] Kamishima, T., Kazawa, H., & Akaho, S. (2005, November).
               "Supervised ordering-an empirical survey.",
               Fifth IEEE International Conference on Data Mining.
        """
        self.normalize = normalize
        self.n_features = n_features
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.tol = tol
        self.logger = logging.getLogger('ERR')
        self.fit_intercept = fit_intercept
        self.random_state = check_random_state(random_state)

    def fit(self, X, Y, **kwargs):
        self.logger.debug('Creating the Dataset')
        X_train, Y_train = complete_linear_regression_dataset(X, Y)
        assert X_train.shape[1] == self.n_features
        self.logger.debug('Finished the Dataset')
        if self.alpha < 1e-3:
            self.model = LinearRegression(normalize=self.normalize, fit_intercept=self.fit_intercept)
            self.logger.info("LinearRegression")
        else:
            if self.l1_ratio >= 0.01:
                self.model = ElasticNet(alpha=self.alpha, l1_ratio=self.l1_ratio, normalize=self.normalize,
                                        tol=self.tol, fit_intercept=self.fit_intercept, random_state=self.random_state)
                self.logger.info("Elastic Net")
            else:
                self.model = Ridge(alpha=self.alpha, normalize=self.normalize,
                                   tol=self.tol,
                                   fit_intercept=self.fit_intercept,
                                   random_state=self.random_state)
                self.logger.info("Ridge")
        self.logger.debug('Finished Creating the model, now fitting started')
        self.model.fit(X_train, Y_train)
        self.weights = self.model.coef_.flatten()
        if (self.fit_intercept):
            self.weights = np.append(self.weights, self.model.intercept_)
        self.logger.debug('Fitting Complete')

    def _predict_scores_fixed(self, X, **kwargs):
        n_instances, n_objects, n_features = X.shape
        self.logger.info(
            "For Test instances {} objects {} features {}".format(n_instances,
                                                                  n_objects,
                                                                  n_features))
        scores = np.empty([n_instances, n_objects])
        for i, data_test in enumerate(X):
            assert data_test.shape[1] == self.n_features
            score = self.model.predict(data_test) * -1
            normalize(np.array(score))
            scores[i] = score
        self.logger.info("Done predicting scores")
        return np.array(scores)

    def predict_scores(self, X, **kwargs):
        return super().predict_scores(X, **kwargs)

    def predict(self, X, **kwargs):
        return super().predict(X, **kwargs)

    def predict_pair(self, a, b, **kwargs):
        score_a = self.model.predict(a, **kwargs) * -1
        score_b = self.model.predict(b, **kwargs) * -1
        return [score_a / (score_a + score_b), score_b / (score_a + score_b)]

    def set_tunable_parameters(self, alpha=0.0, l1_ratio=0.5, tol=1e-4, **point):
        self.tol = tol
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        if len(point) > 0:
            self.logger.warning('This ranking algorithm does not support'
                                ' tunable parameters'
                                ' called: {}'.format(print_dictionary(point)))
