import logging

from sklearn.model_selection import train_test_split

from csrank.core.pairwise_svm import PairwiseSVM
from .choice_functions import ChoiceFunctions
from .util import generate_complete_pairwise_dataset


class PairwiseSVMChoiceFunction(PairwiseSVM, ChoiceFunctions):
    def __init__(self, n_object_features, C=1.0, tol=1e-4, normalize=True,
                 fit_intercept=True, random_state=None, **kwargs):
        """ Create an instance of the PairwiseSVMChoiceFunction model.

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
        .. [1] Theodoros Evgeniou, Massimiliano Pontil, and Olivier Toubia. „A convex optimization approach to modeling consumer heterogeneity in conjoint estimation“.
               In: Marketing Science 26.6 (2007), pp. 805–818 (cit. on p. 18)
           [2] Sebastián Maldonado, Ricardo Montoya, and Richard Weber. „Advanced conjoint analysis using feature selection via support vector machines“.
               In: European Journal of Operational Research 241.2 (2015), pp. 564 –574 (cit. on pp. 19, 20).
        """
        super().__init__(n_object_features=n_object_features, C=C, tol=tol, normalize=normalize,
                         fit_intercept=fit_intercept,
                         random_state=random_state, **kwargs)
        self.logger = logging.getLogger(PairwiseSVMChoiceFunction.__name__)
        self.logger.info("Initializing network with object features {}".format(self.n_object_features))
        self.threshold = 0.5

    def convert_instances(self, X, Y):
        self.logger.debug('Creating the Dataset')
        garbage, garbage, x_train, garbage, y_single = generate_complete_pairwise_dataset(X, Y)
        del garbage
        assert x_train.shape[1] == self.n_object_features
        self.logger.debug('Finished the Dataset with instances {}'.format(x_train.shape[0]))
        return x_train, y_single

    def fit(self, X, Y, tune_size=0.1, thin_thresholds=1, **kwd):
        if tune_size > 0:
            X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=tune_size, random_state=self.random_state)
            try:
                super().fit(X_train, Y_train, **kwd)
            finally:
                self.logger.info('Fitting utility function finished. Start tuning threshold.')
                self.threshold = self._tune_threshold(X_val, Y_val, thin_thresholds=thin_thresholds)
        else:
            super().fit(X, Y, **kwd)
            self.threshold = 0.5

    def _predict_scores_fixed(self, X, **kwargs):
        return super()._predict_scores_fixed(X, **kwargs)

    def predict_scores(self, X, **kwargs):
        return super().predict_scores(X, **kwargs)

    def predict_for_scores(self, scores, **kwargs):
        return ChoiceFunctions.predict_for_scores(self, scores, **kwargs)

    def predict(self, X, **kwargs):
        return super().predict(X, **kwargs)
