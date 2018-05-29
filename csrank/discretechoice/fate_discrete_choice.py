import logging

from csrank.discretechoice.discrete_choice import DiscreteObjectChooser
from csrank.fate_network import FATENetwork


class FATEDiscreteChoiceFunction(FATENetwork, DiscreteObjectChooser):
    def __init__(self, loss_function='categorical_hinge',
                 metrics=['categorical_accuracy', 'top_k_categorical_accuracy'], **kwargs):
        self.loss_function = loss_function
        self.metrics = metrics
        super().__init__(**kwargs)
        self.logger = logging.getLogger(FATEDiscreteChoiceFunction.__name__)

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
