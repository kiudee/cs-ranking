from abc import ABCMeta, abstractmethod

import keras.backend as K

from csrank.callbacks import LRScheduler
from csrank.constants import OBJECT_RANKING
from csrank.util import scores_to_rankings

__all__ = ['ObjectRanker']


class ObjectRanker(metaclass=ABCMeta):
    @property
    def learning_problem(self):
        return OBJECT_RANKING

    @abstractmethod
    def fit(self, X, Y, **kwargs):
        """ Fit the object ranking algorithm to a set of objects X and
        orderings Y of those objects.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_objects, n_features)
            Feature vectors of the objects
        Y : array-like, shape (n_samples, n_objects)
            Orderings of the objects

        Returns
        -------
        self : object
            Returns self.
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, X, **kwargs):
        """ Predict orderings for a given collection of sets of objects.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_objects, n_features)


        Returns
        -------
        Y : array-like, shape (n_samples, n_objects)
            Predicted rankings
        """
        self.logger.debug('Predicting started')

        predicted_scores = self.predict_scores(X, **kwargs)
        self.logger.debug('Predicting scores complete')

        predicted_rankings = scores_to_rankings(predicted_scores)
        self.logger.debug('Predicting ranks complete')
        del predicted_scores
        return predicted_rankings

    @abstractmethod
    def predict_scores(self, X, **kwargs):
        """
        Predict the latent utility scores for each object in X.

        We need to distinguish several cases here:
         * Predict with the non-variadic model on the same ranking size
         * Predict with the non-variadic model on a new ranking size
         * Predict with the variadic model on a known ranking size
         * Predict with the variadic model on a new ranking size
         * Predict on a variadic input

        The simplest solution is creating (a) new model(s) in all of the cases,
        even though it/they might already exist.

         Parameters
         ----------
         X : dict or numpy array
            Dictionary with a mapping from ranking size to numpy arrays
            or a single numpy array of size:
            (n_instances, n_objects, n_features)
         """
        self.logger.info("Predicting scores")

        if isinstance(X, dict):
            scores = dict()
            for ranking_size, x in X.items():
                scores[ranking_size] = self._predict_scores_fixed(x, **kwargs)
        else:
            scores = self._predict_scores_fixed(X, **kwargs)
        return scores

    @abstractmethod
    def _predict_scores_fixed(self, X, **kwargs):
        """ Predict borda scores for a given collection of sets of objects.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_objects, n_features)


        Returns
        -------
        Y : array-like, shape (n_samples, n_objects)
            Returns the borda scores of each of the objects for each of the
            samples.
        """
        raise NotImplementedError

    def __call__(self, X, *args, **kwargs):
        """ Predicts orderings for a new set of points X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_objects, n_features)

        Returns
        -------
        Y : array-like, shape (n_samples, n_objects)
            Predicted orderings"""
        return self.predict(X, **kwargs)

    def set_init_lr_callback(self, callbacks):
        for c in callbacks:
            if isinstance(c, LRScheduler):
                c.initial_lr = K.get_value(self.optimizer.lr)
                self.logger.info("Setting lr {} for {}".format(c.initial_lr, c.__name__))
        return callbacks

    @classmethod
    def __subclasshook__(cls, C):
        if cls is ObjectRanker:
            has_fit = any("fit" in B.__dict__ for B in C.__mro__)
            has_predict = any("predict" in B.__dict__ for B in C.__mro__)
            has_scores = any("predict_scores" in B.__dict__ for B in C.__mro__)
            if has_fit and has_predict and has_scores:
                return True
        return NotImplemented
