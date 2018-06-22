from abc import ABCMeta

from csrank.constants import LABEL_RANKING
from csrank.numpy_util import scores_to_rankings


class LabelRanker(metaclass=ABCMeta):
    @property
    def learning_problem(self):
        return LABEL_RANKING

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
                rankings = scores_to_rankings(score)
                result[n] = rankings
        else:
            result = scores_to_rankings(scores)
        return result
