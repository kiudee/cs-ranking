import logging

from csrank.discretechoice.discrete_choice import ObjectChooser
from csrank.fate_network import FATEObjectRankingCore


class FATEObjectChooser(FATEObjectRankingCore, ObjectChooser):
    def __init__(self, loss_function='categorical_hinge', metrics=None,
                 **kwargs):
        FATEObjectRankingCore.__init__(self, **kwargs)
        self.loss_function = loss_function
        if metrics is None:
            metrics = ['categorical_accuracy']
        self.metrics = metrics
        self.model = None
        self.logger = logging.getLogger(FATEObjectChooser.__name__)

    def predict(self, X, **kwargs):
        scores = self.predict_scores(X, **kwargs)
        if self.is_variadic:
            result = dict()
            for n, s in scores.items():
                result[n] = s.argmax(axis=1)
        else:
            self.logger.info("Predicting chosen object")
            result = scores.argmax(axis=1)
        return result
