import logging

import numpy as np
from sklearn.linear_model import LinearRegression, ElasticNet, Ridge
from sklearn.utils import check_random_state

from csrank.learner import Learner
from csrank.numpy_util import normalize
from csrank.objectranking.object_ranker import ObjectRanker
from csrank.util import print_dictionary
from ..dataset_reader.objectranking.util import \
    complete_linear_regression_dataset

__all__ = ['ExpectedRankRegression']


class ExpectedRankRegression(ObjectRanker, Learner):
    def __init__(self, n_object_features, alpha=0.0, l1_ratio=0.5, tol=1e-4, normalize=True, fit_intercept=True,
                 random_state=None, **kwargs):
        """
            Create an expected rank regression model.

            This model normalizes the ranks to [0, 1] and treats them as regression target. For α = 0 we employ simple
            linear regression. For α > 0 the model becomes ridge regression (when l1_ratio = 0) or elastic net
            (when l1_ratio > 0).

            Parameters
            ----------
            n_object_features : int
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
        self.n_object_features = n_object_features
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.tol = tol
        self.logger = logging.getLogger(ExpectedRankRegression.__name__)
        self.fit_intercept = fit_intercept
        self.random_state = check_random_state(random_state)
        self.weights = None

    def fit(self, X, Y, **kwargs):
        self.logger.debug('Creating the Dataset')
        x_train, y_train = complete_linear_regression_dataset(X, Y)
        assert x_train.shape[1] == self.n_object_features
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
        self.model.fit(x_train, y_train)
        self.weights = self.model.coef_.flatten()
        if self.fit_intercept:
            self.weights = np.append(self.weights, self.model.intercept_)
        self.logger.debug('Fitting Complete')

    def _predict_scores_fixed(self, X, **kwargs):
        n_instances, n_objects, n_features = X.shape
        self.logger.info("For Test instances {} objects {} features {}".format(*X.shape))
        X1 = X.reshape(n_instances * n_objects, n_features)
        scores = n_objects - self.model.predict(X1)
        scores = scores.reshape(n_instances, n_objects)
        scores = normalize(scores)
        self.logger.info("Done predicting scores")
        return scores

    def predict_scores(self, X, **kwargs):
        return super().predict_scores(X, **kwargs)

    def predict_for_scores(self, scores, **kwargs):
        return ObjectRanker.predict_for_scores(self, scores, **kwargs)

    def predict(self, X, **kwargs):
        return super().predict(X, **kwargs)

    def set_tunable_parameters(self, alpha=0.0, l1_ratio=0.5, tol=1e-4, **point):
        self.tol = tol
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        if len(point) > 0:
            self.logger.warning('This ranking algorithm does not support'
                                ' tunable parameters'
                                ' called: {}'.format(print_dictionary(point)))
