import logging

from csrank.fate_network import FATENetwork
from csrank.losses import hinged_rank_loss
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

    def __init__(self, loss_function=hinged_rank_loss, metrics=[zero_one_rank_loss_for_scores_ties], **kwargs):
        self.loss_function = loss_function
        self.metrics = metrics
        super().__init__(**kwargs)
        self.logger = logging.getLogger(FATEObjectRanker.__name__)

    def fit(self, X, Y, **kwd):
        super().fit(X, Y, **kwd)

    def _predict_scores_fixed(self, X, **kwargs):
        return super()._predict_scores_fixed(X, **kwargs)

    def predict_scores(self, X, **kwargs):
        return super().predict_scores(X, **kwargs)

    def predict_for_scores(self, scores, **kwargs):
        return ObjectRanker.predict_for_scores(self, scores, **kwargs)

    def predict(self, X, **kwargs):
        return super().predict(X, **kwargs)

    def clear_memory(self, **kwargs):
        self.logger.info("Clearing memory")
        super().clear_memory(**kwargs)
