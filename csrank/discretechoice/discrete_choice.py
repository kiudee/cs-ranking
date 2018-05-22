from abc import ABCMeta, abstractmethod

from csrank.constants import DISCRETE_CHOICE

__all__ = ['DiscreteObjectChooser']


class DiscreteObjectChooser(metaclass=ABCMeta):
    @property
    def learning_problem(self):
        return DISCRETE_CHOICE

    @abstractmethod
    def fit(self, X, Y, **kwargs):
        """ Fit the object ranking algorithm to a set of objects X and
        choices Y of those objects.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_objects, n_features)
            Feature vectors of the objects
        Y : array-like, shape (n_samples)
            Object IDs of the objects chosen from the set

        Returns
        -------
        self : object
            Returns self.
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, X, **kwargs):
        """ Predict choices for a given collection of sets of objects.

        Parameters
        ----------
        X : dict or numpy array
            Dictionary with a mapping from size of the choice set to numpy arrays
            or a single numpy array of size:
            (n_instances, n_objects, n_features)


        Returns
        -------
        Y : dict or numpy array
            Dictionary with a mapping from size of the choice set to numpy arrays
            or a single numpy array containing discrete choices of size:
            (n_instances, 1)
        """

        self.logger.debug('Predicting started')

        scores = self.predict_scores(X, **kwargs)
        self.logger.debug('Predicting scores complete')

        return self.predict_for_scores(scores)

    def predict_for_scores(self, scores):
        """ Predict choices for a given collection scores for the sets of objects.

        Parameters
        ----------
        scores : dict or numpy array
            Dictionary with a mapping from size of the choice set to numpy arrays
            or a single numpy array of size containing scores of each object of size:
            (n_instances, n_objects)


        Returns
        -------
        Y : dict or numpy array
            Dictionary with a mapping from size of the choice set to numpy arrays
            or a single numpy array containing discrete choices of size:
            (n_instances, 1)
        """
        if isinstance(scores, dict):
            result = dict()
            for n, s in scores.items():
                result[n] = s.argmax(axis=1)

        else:
            result = scores.argmax(axis=1)
        return result

    @abstractmethod
    def predict_scores(self, X, **kwargs):
        """
        Predict the latent utility scores for each object in X.

        We need to distinguish several cases here:
         * Predict with the non-variadic model on the same choice set size
         * Predict with the non-variadic model on a new choice set size
         * Predict with the variadic model on a known choice set size
         * Predict with the variadic model on a new choice set size
         * Predict on a variadic input

        The simplest solution is creating (a) new model(s) in all of the cases,
        even though it/they might already exist.

         Parameters
         ----------
         X : dict or numpy array
            Dictionary with a mapping from ranking size to numpy arrays
            or a single numpy array of size:
            (n_instances, n_objects, n_features)

        Returns
        -------
        Y : dict or numpy array
            Dictionary with a mapping from choice set size to numpy arrays
            or a single numpy array of size:
            (n_instances, n_objects)
            Predicted scores
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
            Returns the scores of each of the objects for each of the samples.
        """
        raise NotImplementedError

    def __call__(self, X, *args, **kwargs):
        """ Predicts choices for a new set of points X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_objects, n_features)

        Returns
        -------
        Y : array-like, shape (n_samples)
            Predicted choices"""
        return self.predict(X, **kwargs)

    @classmethod
    def __subclasshook__(cls, C):
        if cls is DiscreteObjectChooser:
            has_fit = any("fit" in B.__dict__ for B in C.__mro__)
            has_predict = any("predict" in B.__dict__ for B in C.__mro__)
            has_scores = any("predict_scores" in B.__dict__ for B in C.__mro__)
            if has_fit and has_predict and has_scores:
                return True
        return NotImplemented
