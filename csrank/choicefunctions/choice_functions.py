from abc import ABCMeta

import numpy as np

from csrank.constants import CHOICE_FUNCTION
from csrank.metrics_np import f1_measure
from csrank.util import progress_bar

__all__ = ['ChoiceFunctions']


class ChoiceFunctions(metaclass=ABCMeta):

    @property
    def learning_problem(self):
        return CHOICE_FUNCTION

    def predict_for_scores(self, scores, **kwargs):
        """
            Binary choice vector :math:`y` represents the choices amongst the objects in :math:`Q`, such that
            :math:`y(k) = 1` represents that the object :math:`x_k` is chosen and :math:`y(k) = 0` represents it is not
            chosen. Predict choices for the scores for a given collection of sets of objects (query sets).

            Parameters
            ----------
            scores : dict or numpy array
                Dictionary with a mapping from query set size to numpy arrays
                or a single numpy array of size containing scores of each object of size:
                (n_instances, n_objects)


            Returns
            -------
            Y : dict or numpy array
                Dictionary with a mapping from query set size to numpy arrays
                or a single numpy array containing predicted choice vectors of size:
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
        best = f1_measure(Y_val, scores > threshold)
        try:
            for i, p in enumerate(probabilities):
                pred = scores > p
                f1 = f1_measure(Y_val, pred)
                if f1 > best:
                    threshold = p
                    best = f1
                progress_bar(i, len(probabilities), status='Tuning threshold')
        except KeyboardInterrupt:
            self.logger.info("Keyboard interrupted")
        self.logger.info('Tuned threshold, obtained {:.2f} which achieved'
                         ' a micro F1-measure of {:.2f}'.format(threshold, best))
        return threshold
