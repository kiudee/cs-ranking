import logging

from csrank.dataset_reader.objectranking.util import generate_complete_pairwise_dataset
from csrank.objectranking.object_ranker import ObjectRanker
from csrank.ranknet_core import RankNetCore

__all__ = ['RankNet']


class RankNet(RankNetCore, ObjectRanker):
    def __init__(self, loss_function='binary_crossentropy', metrics=['binary_accuracy'],
                 **kwargs):
        super().__init__(loss_function=loss_function, metrics=metrics, **kwargs)
        self.logger = logging.getLogger(RankNet.__name__)
        self.logger.info("Initializing network with object features {}".format(self.n_object_features))

    def convert_instances(self, X, Y):
        self.logger.debug('Creating the Dataset')
        garbage, x1, x2, garbage, y_single = generate_complete_pairwise_dataset(X, Y)
        del garbage
        if x1.shape[0] > self.threshold_instances:
            indices = self.random_state.choice(x1.shape[0], self.threshold_instances, replace=False)
            x1 = x1[indices, :]
            x2 = x2[indices, :]
            y_single = y_single[indices]
        self.logger.debug('Finished the Dataset')
        return x1, x2, y_single

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
        super().clear_memory(**kwargs)
