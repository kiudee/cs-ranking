import logging

from sklearn.preprocessing import LabelBinarizer
import numpy as np
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

    def fit(self, X, Y, epochs=10, callbacks=None, validation_split=0.1, verbose=0, **kwd):
        super().fit(X, Y, epochs, callbacks, validation_split, verbose, **kwd)

    def predict(self, X, **kwargs):
        return DiscreteObjectChooser.predict(X, **kwargs)

    def predict_scores(self, X, **kwargs):
        return super().predict_scores(X, **kwargs)
