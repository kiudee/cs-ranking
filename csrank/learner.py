from abc import ABCMeta
from abc import abstractmethod
import logging

from sklearn.base import BaseEstimator
from skorch import NeuralNet

logger = logging.getLogger(__name__)


def filter_dict_by_prefix(source, prefix):
    result = dict()
    for key in source.keys():
        if key.startswith(prefix):
            key_stripped = key[len(prefix) :]
            result[key_stripped] = source[key]
    return result


class Learner(BaseEstimator, metaclass=ABCMeta):
    def _store_kwargs(self, kwargs, allowed_prefixes):
        """Store kwargs with whitelisted prefixes into this objects attributes.

        Raises an exception if one of the kwargs does not match a whiltelisted prefix.
        """

        def starts_with_legal_prefix(key):
            for prefix in allowed_prefixes:
                if key.startswith(prefix):
                    return True
            return False

        for key in kwargs.keys():
            if not starts_with_legal_prefix(key):
                raise TypeError(
                    f"'__init__() got an unexpected keyword argument '{key}'. Allowed prefixes: {allowed_prefixes}."
                )

        vars(self).update(kwargs)

    def _get_prefix_attributes(self, prefix):
        """Return all attributes of this class that start with a given prefix.

        The prefix is stripped in the result. This can be used to pass on some
        parameters to subclasses.
        """
        return filter_dict_by_prefix(self.__dict__, prefix)

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

    def _pre_fit(self):
        """Perform stateful initialization before fitting.

        This function is for initialization that does not depend on the data,
        but still requires some processing and therefore should not happen in
        __init__. Examples include initialization of optimizers, construction
        of NeuralNetwork layers (if it can be done without knowledge of the
        data) etc.

        You should always call this function before fit, even if you do not
        override it. If you override it, you should call the super method first
        so that general initializations can be inherited.
        """
        pass

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


class SkorchInstanceEstimator(NeuralNet, Learner):
    """Base estimator for torch-based ranking and choice tasks.

    This establishes the basic interface of a cs-ranking learner that is
    compatible with scikit-learn. It is based on an skorch estimator with the
    added assumption that the ``module`` expects the number of features per
    object as a parameter. The ``module`` should then predict a score for each
    object which can later be converted to a prediction (i.e. a ranking, a
    general choice or a discrete choice). To derive a new estimator you should
    therefore override the constructor to set default values for the ``module``
    and the ``criterion`` parameter. You should also override the
    ``predict_for_scores`` function to specify how the scores can be converted
    to the target prediction. You may use one of the existing mixins such as
    ``ObjectRanker`` for that purpose.

    See the documentation of ``skorch.NeuralNet`` for a description of the
    possible parameters.
    """

    def _get_extra_module_parameters(self):
        """Return extra parameters that should be passed to the module.

        You should take care to update the dictionary from the ``super``
        implementation when overriding this function. You usually do not want
        to just discard the parameters that are specified by the super class.
        """
        return {"n_features": self.n_features_}

    def get_params_for(self, prefix):
        """Return the init parameters for an attribute.

        This extends the ``get_params_for`` function from skorch to inject
        custom module parameters. This allows us to pass parameters that do not
        directly correspond to parameters of this estimator while also sticking
        to the scikit-learn estimator API. Overriding this function is
        preferable to than overriding ``initialize_module`` since this function
        does not modify the object's state and we can simply extend the results
        of a ``super`` delegation.
        """
        params = super().get_params_for(prefix)

        if prefix == "module":
            # Explicitly set parameters override the default values.
            defaults = self._get_extra_module_parameters()
            defaults.update(params)
            return defaults
        else:
            return params
        return params

    def fit(self, X, y=None, **fit_params):
        """Fit the estimator to data.

        This derives the number of object features from the data and then
        delegates to ``skorch.NeuralNet.fit``. See the documentation of that
        method for more details.

        Parameters
        ----------
        X : input data
            May take various forms, such as numpy arrays or torch datasets. See
            the documentation of ``skorch.NeuralNet.fit`` for more details.

        y : target data
            May take the same forms as ``x``. This is optional since the target
            data may already be included in the data structure that is passed
            as ``X``. See the documentation of ``skorch.NeuralNet.fit`` for
            more details.

        **fit_params : dict
            Additional fit parameters. See the documentation of
            ``skorch.NeuralNet.fit`` for more details.
        """
        dataset = self.get_dataset(X, y)
        (_n_objects, self.n_features_) = dataset[0][0].shape
        NeuralNet.fit(self, X=dataset, y=None, **fit_params)

    def predict(self, X, **kwargs):
        """Predict targets for inputs.

        This delegates to ``csrank.Learner.predict``. See the documentation of
        that function for details.

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
        return Learner.predict(self, X, **kwargs)

    def _predict_scores_fixed(self, X, **kwargs):
        """Predict scores for a collection of sets of objects of the same size.

        This simply queries the torch module for a prediction on the input
        data, which can then be interpreted as scores.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_objects, n_features)
            The input data.

        Returns
        -------
        Y : array-like, shape (n_samples, n_objects)
            The predicted scores.
        """
        return self.predict_proba(X, **kwargs)
