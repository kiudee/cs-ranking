from abc import ABCMeta

from csrank.constants import DISCRETE_CHOICE
from csrank.dataset_reader.discretechoice.util import convert_to_label_encoding

__all__ = ["DiscreteObjectChooser"]


class DiscreteObjectChooser(metaclass=ABCMeta):
    @property
    def learning_problem(self):
        return DISCRETE_CHOICE

    def predict_for_scores(self, scores):
        """
            Binary discrete choice vector :math:`y` represents the choices amongst the objects in :math:`Q`, such that
            :math:`y(k) = 1` represents that the object :math:`x_k` is chosen and :math:`y(k) = 0` represents
            it is not chosen. For choice to be discrete :math:`\\sum_{x_i \\in Q} y(i) = 1`. Predict discrete choices for
            the scores for a given collection of sets of objects (query sets).

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
                or a single numpy array containing predicted discrete choice vectors of size:
                (n_instances, n_objects)
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
