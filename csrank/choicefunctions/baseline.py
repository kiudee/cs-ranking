import logging
import numpy as np

from csrank.learner import Learner
from .choice_functions import ChoiceFunctions


class AllPositive(ChoiceFunctions, Learner):
    def __init__(self, n_object_features, **kwargs):
        super().__init__(n_object_features=n_object_features, **kwargs)
        self.logger = logging.getLogger(AllPositive.__name__)
        self.logger.info("Initializing network with object features {}".format(self.n_object_features))
        self.threshold = 0.5

    def fit(self, X, Y, **kwd):
        pass

    def _predict_scores_fixed(self, X, Y, **kwargs):
        return np.zeros_like(Y) + Y.mean()

    def predict_scores(self, X, **kwargs):
        return super().predict_scores(X, **kwargs)

    def predict_for_scores(self, scores, **kwargs):
        return np.zeros_like(scores) + 1

    def predict(self, X, **kwargs):
        return super().predict(X, **kwargs)
