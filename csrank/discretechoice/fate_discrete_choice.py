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
        self.logger = logging.getLogger(FATEDiscreteChoiceFunction.__name__)

    def fit(self, X, Y, **kwd):
        FATEObjectRankingCore.fit(self, X, Y, **kwd)

    def predict(self, X, **kwargs):
        return DiscreteObjectChooser.predict(self, X, **kwargs)

    def predict_scores(self, X, **kwargs):
        return DiscreteObjectChooser.predict_scores(self, X, **kwargs)

    def _predict_scores_fixed(self, X, **kwargs):
        return FATEObjectRankingCore._predict_scores_fixed(self, X, **kwargs)

    def predict_for_scores(self, scores, **kwargs):
        return DiscreteObjectChooser.predict_for_scores(self, scores, **kwargs)

    def clear_memory(self, n_objects):
        FATEObjectRankingCore.clear_memory(self, n_objects)
