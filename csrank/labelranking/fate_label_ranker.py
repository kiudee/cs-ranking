import logging

import numpy as np
from sklearn.preprocessing import LabelBinarizer

from csrank.core.fate_network import FATENetwork
from csrank.labelranking.label_ranker import LabelRanker
from csrank.losses import hinged_rank_loss
from csrank.metrics import zero_one_rank_loss_for_scores_ties
from csrank.tensorflow_util import tensorify


class FATELabelRanker(FATENetwork, LabelRanker):
    def __init__(self, loss_function=hinged_rank_loss, metrics=[zero_one_rank_loss_for_scores_ties], **kwargs):
        super().__init__(self, label_ranker=True, loss_function=loss_function, metrics=metrics, **kwargs)
        self.logger = logging.getLogger(FATELabelRanker.__name__)

    def one_hot_encoder_labels(self, X, Y):
        x_trans = []
        for i, x in enumerate(X):
            x = x[None, :]
            x = np.repeat(x, len(Y[i]), axis=0)
            label_binarizer = LabelBinarizer()
            label_binarizer.fit(range(max(Y[i]) + 1))
            b = label_binarizer.transform(Y[i])
            x = np.concatenate((x, b), axis=1)
            x_trans.append(x)
        x_trans = np.array(x_trans)
        return x_trans

    def fit(self, X, Y, callbacks=None, validation_split=0.1, verbose=0,
            **kwargs):
        self.logger.info("Fitting started")
        X = self.one_hot_encoder_labels(X, Y)
        super().fit(X=X, Y=Y, callbacks=callbacks, validation_split=validation_split, verbose=verbose, **kwargs)

    def _predict_scores_fixed(self, X, **kwargs):
        self.logger.info("Predicting scores")
        n_instances, n_objects, n_features = tensorify(X).get_shape().as_list()
        y = []
        for i in range(n_instances):
            y.append(np.arange(n_objects))
        y = np.array(y)
        X = self.one_hot_encoder_labels(X, y)
        return FATENetwork._predict_scores_fixed(self, X, **kwargs)

    def predict_scores(self, X, **kwargs):
        return super().predict_scores(self, X, **kwargs)

    def clear_memory(self, **kwargs):
        self.logger.info("Clearing memory")
        FATENetwork.clear_memory(self, **kwargs)

    def predict_for_scores(self, scores, **kwargs):
        return LabelRanker.predict_for_scores(self, scores, **kwargs)

    def predict(self, X, **kwargs):
        return super().predict(self, X, **kwargs)
