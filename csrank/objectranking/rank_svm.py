import logging

import numpy as np
import sklearn.preprocessing as prep
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.utils import check_random_state

from csrank.objectranking.object_ranker import ObjectRanker
from csrank.tunable import Tunable
from csrank.util import print_dictionary
from ..dataset_reader.objectranking.util import \
    generate_complete_pairwise_dataset

__all__ = ['RankSVM']


class RankSVM(ObjectRanker, Tunable):
    _tunable = None

    def __init__(self, n_features, C=1.0, tol=1e-4, normalize=True,
                 fit_intercept=True, random_state=None, **kwargs):
        """ Create an instance of the RankSVM model.

        Parameters
        ----------
        n_features : int
            Number of features of the object space
        C : float, optional
            Penalty parameter of the error term
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
        .. [1] Joachims, T. (2002, July).
               "Optimizing search engines using clickthrough data.",
               Proceedings of the eighth ACM SIGKDD international conference on
               Knowledge discovery and data mining (pp. 133-142). ACM.
        """
        self.normalize = normalize
        self.n_features = n_features
        self.C = C
        self.tol = tol
        self.logger = logging.getLogger('RankSVM')
        self.random_state = check_random_state(random_state)
        self.threshold_instances = 1000000
        self.fit_intercept = fit_intercept

    def fit(self, X, Y, **kwargs):
        self.logger.debug('Creating the Dataset')
        X_train, garbage, garbage, garbage, Y_single = generate_complete_pairwise_dataset(
            X, Y)
        del garbage
        assert X_train.shape[1] == self.n_features

        self.logger.debug(
            'Finished the Dataset with instances {}'.format(X_train.shape[0]))
        if (X_train.shape[0] > self.threshold_instances):
            self.model = LogisticRegression(C=self.C, tol=self.tol,
                fit_intercept=self.fit_intercept,
                random_state=self.random_state)
        else:
            self.model = LinearSVC(C=self.C, tol=self.tol,
                fit_intercept=self.fit_intercept,
                random_state=self.random_state)

        if (self.normalize):
            scaler = prep.StandardScaler()
            X_train = scaler.fit_transform(X_train)
        self.logger.debug('Finished Creating the model, now fitting started')

        self.model.fit(X_train, Y_single)
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
        scores = []
        for data_test in X:
            assert data_test.shape[1] == self.n_features
            weights = np.array(self.model.coef_)[0]
            try:
                score = np.sum(weights * data_test, axis=1)
            except ValueError:
                score = np.sum(weights[1:] * data_test, axis=1)
            scores.append(score)
        self.logger.info("Done predicting scores")
        return np.array(scores)

    def predict_scores(self, X, **kwargs):
        return super().predict_scores(X, **kwargs)

    def predict(self, X, **kwargs):
        return super().predict(X, **kwargs)

    def predict_pair(self, a, b, **kwargs):
        weights = np.array(self.model.coef_)[0]
        score_a = np.sum(weights * a, axis=1)
        score_b = np.sum(weights * b, axis=1)
        return [score_a / (score_a + score_b), score_b / (score_a + score_b)]

    def set_tunable_parameters(self, C=1.0, tol=1e-4, **point):
        self.tol = tol
        self.C = C
        if len(point) > 0:
            self.logger.warning('This ranking algorithm does not support'
                                ' tunable parameters'
                                ' called: {}'.format(print_dictionary(point)))
