import logging

import numpy as np

from csrank.learner import Learner
from .choice_functions import ChoiceFunctions


class AllPositive(ChoiceFunctions, Learner):
    def __init__(self, **kwargs):
        """
            Baseline assigns the average number of chosen objects in the given choice sets and chooses all the objects.

            :param kwargs: Keyword arguments for the algorithms
        """

        self.logger = logging.getLogger(AllPositive.__name__)
        self.model = None

    def fit(self, X, Y, **kwd):
        pass

    def _predict_scores_fixed(self, X, Y, **kwargs):
        return np.zeros_like(Y) + Y.mean()

    def predict_scores(self, X, Y, **kwargs):
        """
           Predict the utility scores for each object in the collection of set of objects.

            Parameters
            ----------
            X : dict or numpy array
               Dictionary with a mapping from ranking size to numpy arrays
               or a single numpy array of size:
               (n_instances, n_objects, n_features)

           Returns
           -------
           Y : dict or numpy array
               Dictionary with a mapping from ranking size to numpy arrays
               or a single numpy array of size:
               (n_instances, n_objects)
               Predicted scores
        """
        self.logger.info("Predicting scores")

        if isinstance(X, dict):
            scores = dict()
            for ranking_size, x in X.items():
                scores[ranking_size] = self._predict_scores_fixed(x, Y[ranking_size], **kwargs)

        else:
            scores = self._predict_scores_fixed(X, **kwargs)
        return scores

    def predict_for_scores(self, scores, **kwargs):
        if isinstance(scores, dict):
            result = dict()
            for n, score in scores.items():
                result[n] = np.ones_like(score, dtype=int)
        else:
            result = np.ones_like(scores, dtype=int)
        return result

    def predict(self, X, **kwargs):
        return super().predict(X, **kwargs)

    def set_tunable_parameters(self, **point):
        pass
