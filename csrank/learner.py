from abc import ABCMeta, abstractmethod

from csrank.tunable import Tunable


class Learner(Tunable, metaclass=ABCMeta):

    @abstractmethod
    def fit(self, X, Y, **kwargs):
        """
            Fit the preference learning algorithm on the provided set of queries X and preferences Y of those objects.
            The provided queries and corresponding preferences are of a fixed size (numpy arrays).

            Parameters
            ----------
            X : array-like, shape (n_samples, n_objects, n_features)
                Feature vectors of the objects
            Y : array-like, shape (n_samples, n_objects)
                Preferences of the objects in form of rankings or choices
        """
        raise NotImplementedError

    @abstractmethod
    def _predict_scores_fixed(self, X, **kwargs):
        """
            Predict the scores for a given collection of sets of objects of same size.

            Parameters
            ----------
            X : array-like, shape (n_samples, n_objects, n_features)


            Returns
            -------
            Y : array-like, shape (n_samples, n_objects)
                Returns the scores of each of the objects for each of the samples.
        """
        raise NotImplementedError

    def predict_for_scores(self, scores, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def predict_scores(self, X, **kwargs):
        """
            Predict the utility scores for each object in the collection of set of objects called a query set.

            Parameters
            ----------
            X : dict or numpy array
                Dictionary with a mapping from query set size to numpy arrays or a single numpy array of size:
                (n_instances, n_objects, n_features)

            Returns
            -------
            Y : dict or numpy array
                Dictionary with a mapping from query set size to numpy arrays or a single numpy array of size:
                (n_instances, n_objects)
        """
        self.logger.info("Predicting scores")

        if isinstance(X, dict):
            scores = dict()
            for ranking_size, x in X.items():
                n_instances, n_objects, n_features = x.shape
                if "clear_memory" in dir(self):
                    self.clear_memory(n_instances=n_instances, n_objects=n_objects, n_features=n_features)
                scores[ranking_size] = self._predict_scores_fixed(x, **kwargs)

        else:
            scores = self._predict_scores_fixed(X, **kwargs)
        return scores

    @abstractmethod
    def predict(self, X, **kwargs):
        """
            Predict preferences in the form of rankings or choices for a given collection of sets of objects called
            a query set using the function :meth:`.predict_for_scores`.

            Parameters
            ----------
            X : dict or numpy array
                Dictionary with a mapping from the query set size to numpy arrays or a single numpy array of size:
                (n_instances, n_objects, n_features)


            Returns
            -------
            Y : dict or numpy array
                Dictionary with a mapping from the query set size to numpy arrays or a single numpy array containing
                predicted preferences of size:
                (n_instances, n_objects)
        """
        self.logger.debug('Predicting started')

        scores = self.predict_scores(X, **kwargs)
        self.logger.debug('Predicting scores complete')

        return self.predict_for_scores(scores, **kwargs)

    def __call__(self, X, *args, **kwargs):
        """
        Predict preferences in the form of rankings or choices for a given collection of sets of objects.

        Parameters
        ----------
        X : dict or numpy array
            Dictionary with a mapping from ranking size to numpy arrays
            or a single numpy array of size:
            (n_instances, n_objects, n_features)


        Returns
        -------
        Y : Dictionary with a mapping from ranking size to numpy arrays
            or a single numpy array of size:
            (n_instances, n_objects)
            Predicted ranking
        """
        return self.predict(X, **kwargs)

    @classmethod
    def __subclasshook__(cls, C):
        if cls is Learner:
            has_fit = any("fit" in B.__dict__ for B in C.__mro__)
            has_predict = any("predict" in B.__dict__ for B in C.__mro__)
            has_scores = any("predict_scores" in B.__dict__ for B in C.__mro__)
            has_scores_fixed = any("_predict_scores_fixed" in B.__dict__ for B in C.__mro__)
            has_predict_for_scores = any("predict_for_scores" in B.__dict__ for B in C.__mro__)
            if has_fit and has_predict and has_scores and has_scores_fixed and has_predict_for_scores:
                return True
        return NotImplemented
