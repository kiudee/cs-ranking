from abc import ABCMeta

import numpy as np

from csrank.constants import CHOICE_FUNCTIONS

__all__ = ['ChoiceFunctions']


class ChoiceFunctions(metaclass=ABCMeta):

    @property
    def learning_problem(self):
        return CHOICE_FUNCTIONS

    def predict_for_scores(self, scores, **kwargs):
        """ Predict rankings for scores for a given collection of sets of objects.

        Parameters
        ----------
        scores : dict or numpy array
            Dictionary with a mapping from ranking size to numpy arrays
            or a single numpy array of size containing scores of each object of size:
            (n_instances, n_objects)


        Returns
        -------
        Y : dict or numpy array
            Dictionary with a mapping from ranking size to numpy arrays
            or a single numpy array containing predicted ranking of size:
            (n_instances, n_objects)
        """

        if isinstance(scores, dict):
            result = dict()
            for n, score in scores.items():
                result[n] = score > self.threshold
                result[n] = np.array(result[n], dtype=int)
        else:
            result = scores > self.threshold
            result = np.array(result, dtype=int)
        return result
