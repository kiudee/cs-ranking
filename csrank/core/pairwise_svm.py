import logging

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.utils import check_random_state

from csrank.learner import Learner


class PairwiseSVM(Learner):
    def __init__(
        self,
        C=1.0,
        tol=1e-4,
        normalize=True,
        fit_intercept=True,
        use_logistic_regression=False,
        random_state=None,
        **kwargs,
    ):
        """ Create an instance of the PairwiseSVM model for any preference learner.

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
        use_logistic_regression : bool, optional
            Whether to fit a Linear Support Vector machine or a Logistic
            Regression model. You may want to prefer the simpler Logistic
            Regression model on a large sample size.
        random_state : int, RandomState instance or None, optional
            Seed of the pseudorandom generator or a RandomState instance
        **kwargs
            Keyword arguments for the algorithms

        References
        ----------
            [1] Joachims, T. (2002, July). "Optimizing search engines using clickthrough data.", Proceedings of the eighth ACM SIGKDD international conference on Knowledge discovery and data mining (pp. 133-142). ACM.
        """
        self.normalize = normalize
        self.C = C
        self.tol = tol
        self.logger = logging.getLogger("RankSVM")
        self.use_logistic_regression = use_logistic_regression
        self.random_state = random_state
        self.fit_intercept = fit_intercept
        self.weights = None

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
        self.random_state_ = check_random_state(self.random_state)
        _n_instances, self.n_objects_fit_, self.n_object_features_fit_ = X.shape
        x_train, y_single = self._convert_instances_(X, Y)
        if self.use_logistic_regression:
            self.model_ = LogisticRegression(
                C=self.C,
                tol=self.tol,
                fit_intercept=self.fit_intercept,
                random_state=self.random_state_,
            )
            self.logger.info("Logistic Regression model ")
        else:
            self.model_ = LinearSVC(
                C=self.C,
                tol=self.tol,
                fit_intercept=self.fit_intercept,
                random_state=self.random_state_,
            )
            self.logger.info("Linear SVC model ")

        if self.normalize:
            std_scalar = StandardScaler()
            x_train = std_scalar.fit_transform(x_train)
        self.logger.debug("Finished Creating the model, now fitting started")

        self.model_.fit(x_train, y_single)
        self.weights = self.model_.coef_.flatten()
        if self.fit_intercept:
            self.weights = np.append(self.weights, self.model_.intercept_)
        self.logger.debug("Fitting Complete")

    def _predict_scores_fixed(self, X, **kwargs):
        assert X.shape[-1] == self.n_object_features_fit_
        self.logger.info(
            "For Test instances {} objects {} features {}".format(*X.shape)
        )
        if self.fit_intercept:
            scores = np.dot(X, self.weights[:-1])
        else:
            scores = np.dot(X, self.weights)
        self.logger.info("Done predicting scores")
        return np.array(scores)

    def _convert_instances_(self, X, Y):
        raise NotImplementedError
