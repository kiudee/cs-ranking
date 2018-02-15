import logging

from keras.losses import binary_crossentropy

from csrank.fate_ranking import FATEObjectRankingCore


class FATEChoiceFunction(FATEObjectRankingCore):
    def __init__(self,
                 n_hidden_joint_layers=2,
                 n_hidden_joint_units=32,
                 n_hidden_set_layers=2,
                 n_hidden_set_units=32,
                 loss_function=binary_crossentropy,
                 **kwargs):
        super().__init__(n_hidden_joint_layers=n_hidden_joint_layers,
                         n_hidden_joint_units=n_hidden_joint_units,
                         n_hidden_set_layers=n_hidden_set_layers,
                         n_hidden_set_units=n_hidden_set_units,
                         loss_function=loss_function,
                         **kwargs)
        self.logger = logging.Logger('FATEChoiceFunction')

    def fit(self, X, Y, **kwargs):
        super().fit(X, Y, **kwargs)

    def predict_scores(self, X, **kwargs):
        return super().predict_scores(X, **kwargs)

    def predict(self, X, **kwargs):
        scores = self.predict_scores(X, **kwargs)
        return scores > 0.5
