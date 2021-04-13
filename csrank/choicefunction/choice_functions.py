from abc import ABCMeta
import logging

import numpy as np
import skorch
import torch.nn as nn

from csrank.constants import CHOICE_FUNCTION
from csrank.learner import SkorchInstanceEstimator
from csrank.metrics_np import f1_measure
from csrank.util import progress_bar

__all__ = ["ChoiceFunctions"]
logger = logging.getLogger(__name__)


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
                result[n] = score > self.threshold_
                result[n] = np.array(result[n], dtype=int)
        else:
            result = scores > self.threshold_
            result = np.array(result, dtype=int)
        return result

    def _tune_threshold(self, X_val, Y_val, thin_thresholds=1, verbose=0):
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
                if verbose == 1:
                    progress_bar(i, len(probabilities), status="Tuning threshold")
        except KeyboardInterrupt:
            logger.info("Keyboard interrupted")
        logger.info(
            "Tuned threshold, obtained {:.2f} which achieved"
            " a micro F1-measure of {:.2f}".format(threshold, best)
        )
        return threshold


class SkorchChoiceFunction(ChoiceFunctions, SkorchInstanceEstimator):
    """A variable choice estimator based on some scoring module.

    This estimator takes a scoring module and combines it with a sigmoid
    activation to predict scores between 0 and 1. The choice is then made based
    on a fixed threshold value. This makes it very simple to derive new
    estimators with any given scoring function. Refer to skorch's documentation
    for supported parameters. For example the optimizer or the optimizer's
    learning rate could be overridden.

    Parameters
    ----------
    module : torch module (class)
        This is the scoring module. It should be an uninstantiated
        ``torch.nn.Module`` class that expects the number of features per
        object as its only parameter on initialization.

    criterion : torch criterion (class)
        The criterion that is used to evaluate and optimize the module.

    threshold : float
        The threshold value that is used to convert scores to a choice. Must be
        between 0 and 1. Defaults to 0.5.

    **kwargs : skorch NeuralNet arguments
        All keyword arguments are passed to the constructor of
        ``skorch.NeuralNet``. See the documentation of that class for more
        details.
    """

    def __init__(self, module, criterion=nn.BCELoss, threshold=0.5, **kwargs):
        super().__init__(module=module, criterion=criterion, **kwargs)
        # The scoring is trained to predict something close to "0" for
        # non-chosen values, something close to "1" for chosen values. So 0.5
        # is a natural threshold. It would be possible to additionally tune
        # that threshold.
        self.threshold_ = threshold

    def initialize_module(self, *args, **kwargs):
        params = self.get_params_for("module")
        # Add a Sigmoid activation since the resulting "scores" should be
        # between 0 and 1.
        self.module_ = nn.Sequential(self.module(**params), nn.Sigmoid())
        self.module_ = skorch.utils.to_device(self.module_, self.device)
