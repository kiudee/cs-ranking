import logging

from csrank.discretechoice.discrete_choice import DiscreteObjectChooser
from csrank.fate_network import FATENetwork


class FATEDiscreteChoiceFunction(FATENetwork, DiscreteObjectChooser):
    def __init__(self, loss_function='categorical_hinge', metrics=['categorical_accuracy'],
                 **kwargs):
        FATENetwork.__init__(self, loss_function=loss_function, metrics=metrics ** kwargs)
        self.logger = logging.getLogger(FATEDiscreteChoiceFunction.__name__)

    def fit(self, X, Y, **kwd):
        super().fit(self, X, Y, **kwd)

    def predict_scores(self, X, **kwargs):
        return super().predict_scores(X, **kwargs)

    def predict_for_scores(self, scores, **kwargs):
        return DiscreteObjectChooser.predict_for_scores(self, scores, **kwargs)

    def predict(self, X, **kwargs):
        return super().predict(self, X, **kwargs)

    def _predict_scores_fixed(self, X, **kwargs):
        return super()._predict_scores_fixed(self, X, **kwargs)

    def clear_memory(self, **kwargs):
        super().clear_memory(self, **kwargs)