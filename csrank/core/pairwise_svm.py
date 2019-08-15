import logging

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.utils import check_random_state

from csrank.learner import Learner
from csrank.util import print_dictionary


class PairwiseSVM(Learner):
    def __init__(self, n_object_features, C=1.0, tol=1e-4, normalize=True, fit_intercept=True, random_state=None,
                 **kwargs):
        """ Create an instance of the PairwiseSVM model for any preference learner.

        Parameters
        ----------
        n_object_features : int
            Number of features of the object space
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
        self.normalize = normalize
        self.n_object_features = n_object_features
        self.C = C
        self.tol = tol
        self.logger = logging.getLogger('RankSVM')
        self.random_state = check_random_state(random_state)
        self.threshold_instances = int(1e10)
        self.fit_intercept = fit_intercept
        self.weights = None
        self.model = None

    def fit(self, X, Y, **kwargs):
        """
            Fit a generic preference learning model on a provided set of queries.
            The provided queries can be of a fixed size (numpy arrays).

            Parameters
            ----------
            X : numpy array, shape (n_samples, n_objects, n_features)
                Feature vectors of the objects
            Y : numpy array, shape (n_samples, n_objects, n_features)
                Preferences in form of Orderings or Choices for given n_objects
            **kwargs
                Keyword arguments for the fit function

        """
        x_train, y_single = self._convert_instances_(X, Y)
        if x_train.shape[0] > self.threshold_instances:
            self.model = LogisticRegression(C=self.C, tol=self.tol, fit_intercept=self.fit_intercept,
                                            random_state=self.random_state)
            self.logger.info("Logistic Regression model ")
        else:
            self.model = LinearSVC(C=self.C, tol=self.tol, fit_intercept=self.fit_intercept,
                                   random_state=self.random_state)
            self.logger.info("Linear SVC model ")

        if self.normalize:
            std_scalar = StandardScaler()
            x_train = std_scalar.fit_transform(x_train)
        self.logger.debug('Finished Creating the model, now fitting started')

        self.model.fit(x_train, y_single)
        self.weights = self.model.coef_.flatten()
        if self.fit_intercept:
            self.weights = np.append(self.weights, self.model.intercept_)
        self.logger.debug('Fitting Complete')

    def _predict_scores_fixed(self, X, **kwargs):
        assert X.shape[-1] == self.n_object_features
        self.logger.info("For Test instances {} objects {} features {}".format(*X.shape))
        if self.fit_intercept:
            scores = np.dot(X, self.weights[:-1])
        else:
            scores = np.dot(X, self.weights)
        self.logger.info("Done predicting scores")
        return np.array(scores)

    def _convert_instances_(self, X, Y):
        raise NotImplemented

    def set_tunable_parameters(self, C=1.0, tol=1e-4, **point):
        """
            Set tunable parameters of the PairwiseSVM model to the values provided.

            Parameters
            ----------
            C : float
               Penalty parameter of the error term of the SVM classifier
            tol : float
                Optimization tolerance of the SVM classifier
            point: dict
                Dictionary containing parameter values which are not tuned for the network
        """
        self.tol = tol
        self.C = C
        if len(point) > 0:
            self.logger.warning('This ranking algorithm does not support'
                                ' tunable parameters'
                                ' called: {}'.format(print_dictionary(point)))
