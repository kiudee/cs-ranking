from abc import ABCMeta
from abc import abstractmethod

from csrank.constants import DYAD_RANKING

__all__ = ["DyadRanker"]


class DyadRanker(metaclass=ABCMeta):
    @property
    def learning_problem(self):
        return DYAD_RANKING

    @abstractmethod
    def fit(self, Xo, Xc, Y, **kwargs):
        """Fit the contextual object ranking algorithm for object features
         Xo and context features Xc and orderings Y for the objects.

        Parameters
        ----------
        Xo : array-like, shape (n_samples, n_objects, n_object_features)
            Feature vectors of the objects
        Xc : array-like, shape (n_samples, n_context_features)
            Feature vector of the context
        Y : array-like, shape (n_samples, n_objects)
            Orderings of the objects

        Returns
        -------
        self : object
            Returns self.
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, Xo, Xc, **kwargs):
        """Predict orderings for the objects for a given context vector.

        Parameters
        ----------
        Xo : array-like, shape (n_samples, n_objects, n_object_features)
        Xc : array-like, shape (n_samples, n_context_features)


        Returns
        -------
        Y : array-like, shape (n_samples, n_objects)
            Predicted orderings
        """
        raise NotImplementedError

    @abstractmethod
    def predict_scores(self, Xo, Xc, **kwargs):
        """Predict scores for a given collection of sets of objects.

        Parameters
        ----------
        Xo : array-like, shape (n_samples, n_objects, n_object_features)
        Xc : array-like, shape (n_samples, n_context_features)


        Returns
        -------
        Y : array-like, shape (n_samples, n_objects)
            Returns the scores of each of the objects for each of the
            samples.
        """
        raise NotImplementedError

    def __call__(self, Xo, Xc, *args, **kwargs):
        """Predicts orderings for a new set of objects Xo.

        Parameters
        ----------
        Xo : array-like, shape (n_samples, n_objects, n_object_features)
        Xc : array-like, shape (n_samples, n_context_features)

        Returns
        -------
        Y : array-like, shape (n_samples, n_objects)
            Predicted orderings"""
        return self.predict(Xo, Xc, **kwargs)

    @classmethod
    def __subclasshook__(cls, C):
        if cls is DyadRanker:
            has_fit = any("fit" in B.__dict__ for B in C.__mro__)
            has_predict = any("predict" in B.__dict__ for B in C.__mro__)
            has_scores = any("predict_scores" in B.__dict__ for B in C.__mro__)
            if has_fit and has_predict and has_scores:
                return True
        return NotImplemented
