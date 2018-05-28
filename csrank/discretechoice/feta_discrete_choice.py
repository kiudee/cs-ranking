import logging

from csrank.feta_network import FETANetwork
from .discrete_choice import DiscreteObjectChooser


class FETADiscreteChoiceFunction(FETANetwork, DiscreteObjectChooser):
    def __init__(self, loss_function='categorical_hinge', metrics=['categorical_accuracy'], **kwargs):
        super().__init__(metrics=metrics, loss_function=loss_function, **kwargs)
        self.logger = logging.getLogger(FETADiscreteChoiceFunction.__name__)

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
