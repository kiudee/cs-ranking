import logging

from csrank.fate_network import FATENetwork
from csrank.losses import smooth_rank_loss
from csrank.metrics import zero_one_rank_loss_for_scores_ties
from csrank.objectranking.object_ranker import ObjectRanker


class FATEObjectRanker(FATENetwork, ObjectRanker):
    """ Create a FATE-network architecture for object ranking.

        Training complexity is quadratic in the number of objects and
        prediction complexity is only linear.

        Parameters
        ----------
        loss_function : function
            Differentiable loss function for the score vector
        metrics : list
            List of evaluation metrics (can be non-differentiable)
        **kwargs
            Keyword arguments for the @FATENetwork
        """

    def __init__(self, loss_function=smooth_rank_loss, metrics=None, **kwargs):
        if metrics is None:
            metrics = [zero_one_rank_loss_for_scores_ties]
        FATENetwork.__init__(self, loss_function=loss_function, metrics=metrics, **kwargs)
        self.logger = logging.getLogger(FATEObjectRanker.__name__)

    def fit(self, X, Y, **kwd):
        super().fit(self, X, Y, **kwd)

    def predict(self, X, **kwargs):
        return super().predict(self, X, **kwargs)

    def predict_for_scores(self, scores, **kwargs):
        ObjectRanker.predict_for_scores(self, scores, **kwargs)

    def _predict_scores_fixed(self, X, **kwargs):
        return super()._predict_scores_fixed(self, X, **kwargs)

    def predict_scores(self, X, **kwargs):
        return super().predict_scores(self, X, **kwargs)

    def clear_memory(self, n_objects):
        self.logger.info("Clearing memory")
        FATENetwork.clear_memory(self, n_objects)
