import logging

from csrank.dataset_reader.discretechoice.util import generate_complete_pairwise_dataset
from csrank.ranknet_core import RankNetCore
from .discrete_choice import DiscreteObjectChooser


class RankNetDiscreteChoiceFunction(RankNetCore, DiscreteObjectChooser):
    def __init__(self, loss_function='binary_crossentropy', metrics=['accuracy'], **kwargs):
        super().__init__(loss_function=loss_function, metrics=metrics, **kwargs)
        self.logger = logging.getLogger(RankNetDiscreteChoiceFunction.__name__)

    def convert_instances(self, X, Y):
        self.logger.debug('Creating the Dataset')
        X1, X2, garbage, Y_single = generate_complete_pairwise_dataset(X, Y)
        del garbage
        if X1.shape[0] > self.threshold_instances:
            indices = self.random_state.choice(X1.shape[0], self.threshold_instances, replace=False)
            X1 = X1[indices, :]
            X2 = X2[indices, :]
            Y_single = Y_single[indices]
        self.logger.debug('Finished the Dataset')
        return X1, X2, Y_single

    def fit(self, X, Y, **kwd):
        super().fit(X, Y, **kwd)

    def _predict_scores_fixed(self, X, **kwargs):
        return super()._predict_scores_fixed(X, **kwargs)

    def predict_scores(self, X, **kwargs):
        return super().predict_scores(X, **kwargs)

    def predict_for_scores(self, scores, **kwargs):
        return DiscreteObjectChooser.predict_for_scores(self, scores, **kwargs)

    def predict(self, X, **kwargs):
        return super().predict(X, **kwargs)

    def clear_memory(self, **kwargs):
        self.logger.info("Clearing memory")
        super().clear_memory(**kwargs)
