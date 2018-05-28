from abc import ABCMeta

from csrank.constants import DISCRETE_CHOICE
from csrank.dataset_reader.util import convert_to_label_encoding

__all__ = ['DiscreteObjectChooser']


class DiscreteObjectChooser(metaclass=ABCMeta):

    @property
    def learning_problem(self):
        return DISCRETE_CHOICE

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
                result[n] = convert_to_label_encoding(result[n], n)

        else:
            n = scores.shape[-1]
            result = scores.argmax(axis=1)
            result = convert_to_label_encoding(result, n)
        return result
