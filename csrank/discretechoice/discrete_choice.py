from abc import ABCMeta

from csrank.constants import DISCRETE_CHOICE
from csrank.dataset_reader.discretechoice.util import convert_to_label_encoding
from csrank.discrete_choice_losses import CategoricalHingeLossMax
from csrank.learner import SkorchInstanceEstimator

__all__ = ["DiscreteObjectChooser", "SkorchDiscreteChoiceFunction"]


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


class SkorchDiscreteChoiceFunction(DiscreteObjectChooser, SkorchInstanceEstimator):
    """Base estimator for torch-based discrete choice.

    This makes it very simple to derive new estimators with any given scoring
    module. Refer to skorch's documentation for supported parameters. For
    example the optimizer or the optimizer's learning rate could be overridden.

    Parameters
    ----------
    module : torch module (class)
        This is the scoring module. It should be an uninstantiated
        ``torch.nn.Module`` class that expects the number of features per
        object as its only parameter on initialization.

    criterion : torch criterion (class)
        The criterion that is used to evaluate and optimize the module.

    choice_size : int
        The size of the target choice set.

    **kwargs : skorch NeuralNet arguments
        All keyword arguments are passed to the constructor of
        ``skorch.NeuralNet``. See the documentation of that class for more
        details.
    """

    def __init__(
        self, module, criterion=CategoricalHingeLossMax, choice_size=1, **kwargs
    ):
        super().__init__(module=module, criterion=criterion, **kwargs)
        self.choice_size = choice_size
