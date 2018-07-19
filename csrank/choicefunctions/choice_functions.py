from abc import ABCMeta

import numpy as np
from sklearn.metrics import f1_score

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

    def _tune_threshold(self, X_val, Y_val, thin_thresholds=1):
        scores = self.predict_scores(X_val)
        probabilities = np.unique(scores)[::thin_thresholds]
        threshold = 0.0
        best = f1_score(Y_val, scores > threshold, average='samples')
        for i, p in enumerate(probabilities):
            pred = scores > p
            f1 = f1_score(Y_val, pred, average='samples')
            if f1 > best:
                threshold = p
                best = f1
        self.logger.info('Tuned threshold, obtained {:.2f} which achieved'
                         ' a micro F1-measure of {:.2f}'.format(threshold, best))
        return threshold
