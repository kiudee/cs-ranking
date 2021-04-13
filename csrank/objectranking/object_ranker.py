from abc import ABCMeta

from csrank.constants import OBJECT_RANKING
from csrank.learner import SkorchInstanceEstimator
from csrank.numpy_util import scores_to_rankings
from csrank.rank_losses import HingedRankLoss

__all__ = ["ObjectRanker", "SkorchObjectRanker"]


class ObjectRanker(metaclass=ABCMeta):
    @property
    def learning_problem(self):
        return OBJECT_RANKING

    def predict_for_scores(self, scores, **kwargs):
        """
        The permutation vector :math:`\\pi` represents the ranking amongst the objects in :math:`Q`, such that
        :math:`\\pi(k)` is the position of the :math:`k`-th object :math:`x_k`, and :math:`\\pi^{-1}(k)` is the index
        of the object on position :math:`k`. Predict rankings for the scores for a given collection of sets of
        objects (query sets).

        Parameters
        ----------
        scores : dict or numpy array
            Dictionary with a mapping from query set size to numpy arrays or a single numpy array of size containing
            scores of each object of size: (n_instances, n_objects)

        Returns
        -------
        Y : dict or numpy array
            Dictionary with a mapping from objects size to numpy arrays or a single numpy array containing
            predicted rankings of size: (n_samples, n_objects)
        """

        if isinstance(scores, dict):
            result = dict()
            for n, score in scores.items():
                rankings = scores_to_rankings(score)
                result[n] = rankings
        else:
            result = scores_to_rankings(scores)
        return result


class SkorchObjectRanker(ObjectRanker, SkorchInstanceEstimator):
    """Base estimator for torch-based ranking.

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

    **kwargs : skorch NeuralNet arguments
        All keyword arguments are passed to the constructor of
        ``skorch.NeuralNet``. See the documentation of that class for more
        details.
    """

    def __init__(self, module, criterion=HingedRankLoss, **kwargs):
        super().__init__(module=module, criterion=criterion, **kwargs)
