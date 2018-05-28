import logging

import numpy as np
from keras.layers import Dense
from keras.losses import binary_crossentropy
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from csrank.fate_network import FATENetwork
from csrank.metrics import zero_one_accuracy
from .choice_functions import ChoiceFunctions


class FATEChoiceFunction(FATENetwork, ChoiceFunctions):
    def __init__(self, loss_function=binary_crossentropy, metrics=None, **kwargs):
        if metrics is None:
            metrics = [zero_one_accuracy]
        super().__init__(self, loss_function=loss_function, metrics=metrics, **kwargs)
        self.logger = logging.Logger(FATEChoiceFunction.__name__)
        self.threshold = 0.5

    def _construct_layers(self, **kwargs):
        """ Construct joint layers and [0,1] output nodes

        Connecting the layers is done in join_input_layers.

        Parameters
        ----------
        **kwargs
            Keyword arguments passed into the joint layers
        """
        self.logger.info(
            "Construct joint layers hidden units {} and layers {} ".format(
                self.n_hidden_joint_units,
                self.n_hidden_joint_layers))
        # Create joint hidden layers:
        self.joint_layers = []
        for i in range(self.n_hidden_joint_layers):
            self.joint_layers.append(Dense(self.n_hidden_joint_units, name="joint_layer_{}".format(i), **kwargs))
        self.logger.info('Construct output score node')
        self.scorer = Dense(1, name="output_node", activation='sigmoid', kernel_regularizer=self.kernel_regularizer)

    def _tune_threshold(self, X_val, Y_val, thin_thresholds=1):
        scores = self.predict_scores(X_val)
        probabilities = np.unique(scores)[::thin_thresholds]
        threshold = 0.0
        best = f1_score(Y_val, scores > threshold, average='samples')
        for i, p in enumerate(probabilities):
            predictions = scores > p
            f1 = f1_score(Y_val, predictions, average='samples')
            if f1 > best:
                threshold = p
                best = f1
        self.logger.info(
            'Tuned threshold, obtained {:.2f} which achieved a micro F1-measure of {:.2f}'.format(threshold, best))
        return threshold

    def fit(self, X, Y, tune_size=0.1, thin_thresholds=1, **kwargs):
        if tune_size > 0:
            X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=tune_size)
            try:
                super().fit(X_train, Y_train, **kwargs)
            finally:
                self.logger.info('Fitting utility function finished. Start tuning threshold.')
                self.threshold = self._tune_threshold(X_val, Y_val, thin_thresholds=thin_thresholds)
        else:
            super().fit(X, Y, **kwargs)
            self.threshold = 0.5

    def predict_scores(self, X, **kwargs):
        return super().predict_scores(X, **kwargs)

    def predict_for_scores(self, scores, **kwargs):
        return ChoiceFunctions.predict_for_scores(self, scores, **kwargs)

    def predict(self, X, **kwargs):
        return super().predict(self, X, **kwargs)

    def _predict_scores_fixed(self, X, **kwargs):
        return super()._predict_scores_fixed(self, X, **kwargs)

    def clear_memory(self, **kwargs):
        super().clear_memory(self, **kwargs)
