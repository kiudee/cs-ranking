import logging
import numpy as np

from csrank.learner import Learner
from .choice_functions import ChoiceFunctions


class AllPositive(ChoiceFunctions, Learner):
    def __init__(self, **kwargs):

        self.logger = logging.getLogger(AllPositive.__name__)

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
