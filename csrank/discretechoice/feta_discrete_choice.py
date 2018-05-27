import logging

from csrank.objectranking.feta_ranker import FETAObjectRanker
from .discrete_choice import DiscreteObjectChooser


class FETADiscreteChoiceFunction(FETAObjectRanker, DiscreteObjectChooser):
    def __init__(self, loss_function='categorical_hinge', metrics=None, **kwargs):
        FETAObjectRanker.__init__(self, **kwargs)
        self.loss_function = loss_function
        if metrics is None:
            metrics = ['categorical_accuracy']
        self.metrics = metrics
        self.logger = logging.getLogger(FETADiscreteChoiceFunction.__name__)

    def fit(self, X, Y, **kwd):
        FETAObjectRanker.fit(self, X, Y, **kwd)

    def predict_scores(self, X, **kwargs):
        return DiscreteObjectChooser.predict_scores(self, X, **kwargs)

    def predict(self, X, **kwargs):
        return DiscreteObjectChooser.predict(self, X, **kwargs)

    def _predict_scores_fixed(self, X, **kwargs):
        return FETAObjectRanker._predict_scores_fixed(self, X, **kwargs)

    def predict_for_scores(self, scores, **kwargs):
        return DiscreteObjectChooser.predict_for_scores(self, scores, **kwargs)

    def clear_memory(self, n_objects):
        FETAObjectRanker.clear_memory(self, n_objects)
