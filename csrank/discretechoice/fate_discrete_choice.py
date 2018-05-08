import logging

from csrank.discretechoice.discrete_choice import DiscreteObjectChooser
from csrank.fate_network import FATEObjectRankingCore


class FATEDiscreteChoiceFunction(FATEObjectRankingCore, DiscreteObjectChooser):
    def __init__(self, loss_function='categorical_hinge', metrics=None,
                 **kwargs):
        FATEObjectRankingCore.__init__(self, **kwargs)
        self.loss_function = loss_function
        if metrics is None:
            metrics = ['categorical_accuracy']
        self.metrics = metrics
        self.model = None
        self.logger = logging.getLogger(FATEDiscreteChoiceFunction.__name__)

    def predict(self, X, **kwargs):
        return DiscreteObjectChooser.predict(X, **kwargs)

    def predict_scores(self, X, **kwargs):
        return super().predict_scores(X, **kwargs)
