import logging

from csrank.fate_network import FATEObjectRankingCore
from csrank.losses import smooth_rank_loss
from csrank.metrics import zero_one_rank_loss_for_scores_ties, zero_one_rank_loss_for_scores
from csrank.objectranking.object_ranker import ObjectRanker
from csrank.util import scores_to_rankings


class FATEObjectRanker(FATEObjectRankingCore, ObjectRanker):
    """ Create a FATE-network architecture for object ranking.

        Training complexity is quadratic in the number of objects and
        prediction complexity is only linear.

        Parameters
        ----------
        n_object_features : int
            Dimensionality of the feature space of each object
        n_hidden_set_layers : int
            Number of hidden layers for the context representation
        n_hidden_set_units : int
            Number of hidden units in each layer of the context representation
        loss_function : function
            Differentiable loss function for the score vector
        metrics : list
            List of evaluation metrics (can be non-differentiable)
        **kwargs
            Keyword arguments for the hidden units
        """

    def __init__(self, n_object_features,
                 n_hidden_set_layers=2,
                 n_hidden_set_units=32,
                 loss_function=smooth_rank_loss,
                 metrics=None,
                 **kwargs):
        FATEObjectRankingCore.__init__(self,
            n_object_features=n_object_features,
            n_hidden_set_layers=n_hidden_set_layers,
            n_hidden_set_units=n_hidden_set_units,
            **kwargs)
        self.loss_function = loss_function
        self.logger = logging.getLogger(FATEObjectRanker.__name__)
        if metrics is None:
            metrics = [zero_one_rank_loss_for_scores_ties,
                       zero_one_rank_loss_for_scores]
        self.metrics = metrics
        self.logger.info("Initializing network with object features {}".format(
            self.n_object_features))

    def predict(self, X, **kwargs):
        self.logger.info("Predicting ranks")
        if isinstance(X, dict):
            result = dict()
            for n, scores in self.predict_scores(X, **kwargs).items():
                predicted_rankings = scores_to_rankings(scores)
                result[n] = predicted_rankings
            return result
        return ObjectRanker.predict(self, X, **kwargs)

    def _predict_scores_fixed(self, X, **kwargs):
        return FATEObjectRankingCore._predict_scores_fixed(self, X, **kwargs)
