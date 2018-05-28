import logging

from csrank.feta_network import FETANetwork
from .discrete_choice import DiscreteObjectChooser


class FETADiscreteChoiceFunction(FETANetwork, DiscreteObjectChooser):
    def __init__(self, loss_function='categorical_hinge', metrics=['categorical_accuracy'], **kwargs):
        super().__init__(self, metrics=metrics, loss_function=loss_function, **kwargs)
        self.logger = logging.getLogger(FETADiscreteChoiceFunction.__name__)

    def fit(self, X, Y, **kwd):
        super().fit(self, X, Y, **kwd)

    def predict(self, X, **kwargs):
        return super().predict(self, X, **kwargs)

    def predict_for_scores(self, scores, **kwargs):
        DiscreteObjectChooser.predict_for_scores(self, scores, **kwargs)

    def _predict_scores_fixed(self, X, **kwargs):
        return super()._predict_scores_fixed(self, X, **kwargs)

    def predict_scores(self, X, **kwargs):
        return super().predict_scores(self, X, **kwargs)

    def clear_memory(self, **kwargs):
        self.logger.info("Clearing memory")
        super().clear_memory(self, **kwargs)
