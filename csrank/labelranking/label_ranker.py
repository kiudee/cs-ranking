from abc import ABCMeta, abstractmethod

from csrank.constants import LABEL_RANKING
from csrank.util import scores_to_rankings


class LabelRanker(metaclass=ABCMeta):
    @property
    def learning_problem(self):
        return LABEL_RANKING

    @abstractmethod
    def fit(self, X, Y, **kwargs):
        """ Fit the label ranking algorithm for a context vector X and
        orderings Y for labels.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature vectors
        Y : array-like, shape (n_samples, n_labels)
            Orderings of the labels

        Returns
        -------
        self : object
            Returns self.
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, X, **kwargs):
        """ Predict orderings for the labels for a given context vector.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)


        Returns
        -------
        Y : array-like, shape (n_samples, n_labels)
            Predicted orderings
        """
        self.logger.debug('Predicting started')

        predicted_scores = self.predict_scores(X, **kwargs)
        self.logger.debug('Predicting scores complete')

        predicted_orderings = scores_to_rankings(predicted_scores)
        self.logger.debug('Predicting ranks complete')

        del predicted_scores
        return predicted_orderings

    @abstractmethod
    def predict_scores(self, X, **kwargs):
        """ Predict scores for all labels on all instances X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)


        Returns
        -------
        Y : array-like, shape (n_samples, n_labels)
            Returns the scores of each of the labels for each of the
            samples.
        """
        raise NotImplementedError

    def __call__(self, X, *args, **kwargs):
        """ Predicts orderings for a new set of points X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        Y : array-like, shape (n_samples, n_labels)
            Predicted orderings"""
        return self.predict(X, **kwargs)

    @classmethod
    def __subclasshook__(cls, C):
        if cls is LabelRanker:
            has_fit = any("fit" in B.__dict__ for B in C.__mro__)
            has_predict = any("predict" in B.__dict__ for B in C.__mro__)
            has_scores = any("predict_scores" in B.__dict__ for B in C.__mro__)
            if has_fit and has_predict and has_scores:
                return True
        return NotImplemented
