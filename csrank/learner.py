from abc import ABCMeta
from abc import abstractmethod
import logging

from sklearn.base import BaseEstimator

logger = logging.getLogger(__name__)


def filter_dict_by_prefix(source, prefix):
    result = dict()
    for key in source.keys():
        if key.startswith(prefix):
            key_stripped = key[len(prefix) :]
            result[key_stripped] = source[key]
    return result


class Learner(BaseEstimator, metaclass=ABCMeta):
    def _initialize_optimizer(self):
        optimizer_params = filter_dict_by_prefix(self.__dict__, "optimizer__")
        optimizer_params.update(filter_dict_by_prefix(self.kwargs, "optimizer__"))
        self.optimizer_ = self.optimizer(**optimizer_params)

    def _initialize_regularizer(self):
        regularizer_params = filter_dict_by_prefix(
            self.__dict__, "kernel_regularizer__"
        )
        regularizer_params.update(
            filter_dict_by_prefix(self.kwargs, "kernel_regularizer__")
        )
        if self.kernel_regularizer is not None:
            print(f"Regularizer is {self.kernel_regularizer}")
            print(f"Initializing with {regularizer_params}")
            self.kernel_regularizer_ = self.kernel_regularizer(**regularizer_params)
        else:
            # No regularizer is an option.
            logger.warning("You specified regularizer parameters but no regularizer.")
            self.kernel_regularizer_ = None

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
        logger.info("Predicting scores")

        if isinstance(X, dict):
            scores = dict()
            for ranking_size, x in X.items():
                n_instances, n_objects, n_features = x.shape
                scores[ranking_size] = self._predict_scores_fixed(x, **kwargs)

        else:
            scores = self._predict_scores_fixed(X, **kwargs)
        return scores

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
        logger.debug("Predicting started")

        scores = self.predict_scores(X, **kwargs)
        logger.debug("Predicting scores complete")

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
            has_scores_fixed = any(
                "_predict_scores_fixed" in B.__dict__ for B in C.__mro__
            )
            has_predict_for_scores = any(
                "predict_for_scores" in B.__dict__ for B in C.__mro__
            )
            if (
                has_fit
                and has_predict
                and has_scores
                and has_scores_fixed
                and has_predict_for_scores
            ):
                return True
        return NotImplemented
