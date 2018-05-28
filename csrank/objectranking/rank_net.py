import logging

from keras.losses import binary_crossentropy
from keras.metrics import top_k_categorical_accuracy, binary_accuracy

from csrank.objectranking.object_ranker import ObjectRanker
from csrank.rank_network import RankNetwork

__all__ = ['RankNet']


class RankNet(RankNetwork, ObjectRanker):
    def __init__(self, loss_function=binary_crossentropy, metrics=[top_k_categorical_accuracy, binary_accuracy],
                 **kwargs):
        RankNetwork.__init__(self, loss_function=loss_function, metrics=metrics, **kwargs)
        self.logger = logging.getLogger(RankNet.__name__)
        self.logger.info("Initializing network with object features {}".format(
            self.n_object_features))

    def predict_scores(self, X, **kwargs):
        return super().predict_scores(X, **kwargs)

    def predict_for_scores(self, scores, **kwargs):
        return ObjectRanker.predict_for_scores(self,scores,**kwargs)

    def predict(self, X, **kwargs):
        return super().predict(self, X,**kwargs)

    def _predict_scores_fixed(self, X, **kwargs):
        return super()._predict_scores_fixed(self, X, **kwargs)

    def clear_memory(self, n_objects):
        self.logger.info("Clearing memory")
        RankNetwork.clear_memory(self, n_objects)
