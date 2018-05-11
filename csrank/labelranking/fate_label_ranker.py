import logging

import numpy as np
from keras import Model
from sklearn.preprocessing import LabelBinarizer

from csrank.fate_network import FATERankingCore, FATEObjectRankingCore
from csrank.labelranking.label_ranker import LabelRanker
from csrank.losses import hinged_rank_loss
from csrank.metrics import zero_one_rank_loss_for_scores_ties, zero_one_rank_loss_for_scores
from csrank.util import tensorify


class FATELabelRanker(FATERankingCore, LabelRanker):
    def __init__(self, loss_function=hinged_rank_loss, metrics=None,
                 **kwargs):
        super().__init__(self, label_ranker=True, **kwargs)
        self.loss_function = loss_function
        self.logger = logging.getLogger(FATELabelRanker.__name__)
        if metrics is None:
            metrics = [zero_one_rank_loss_for_scores_ties, zero_one_rank_loss_for_scores]
        self.metrics = metrics
        self.model = None
        self.logger.info("Initializing network with object features {}".format(self.n_object_features))
        self._connect_layers()

    def one_hot_encoder_lr_data_conversion(self, X, Y):
        X_trans = []
        for i, x in enumerate(X):
            x = x[None, :]
            x = np.repeat(x, len(Y[i]), axis=0)
            label_binarizer = LabelBinarizer()
            label_binarizer.fit(range(max(Y[i]) + 1))
            b = label_binarizer.transform(Y[i])
            x = np.concatenate((x, b), axis=1)
            X_trans.append(x)
        X_trans = np.array(X_trans)
        return X_trans

    def _create_set_layers(self, **kwargs):
        FATEObjectRankingCore._create_set_layers(self, **kwargs)

    def _connect_layers(self):
        self.set_input_layers(self.inputs, self.set_repr, self.n_hidden_set_layers)

    def fit(self, X, Y, callbacks=None, validation_split=0.1, verbose=0,
            **kwargs):
        self.logger.info("Fitting started")
        X_trans = self.one_hot_encoder_lr_data_conversion(X, Y)

        self.model = Model(inputs=self.input_layer, outputs=self.scores)
        self.model.compile(loss=self.loss_function, optimizer=self.optimizer,
                           metrics=self.metrics)
        self.model.fit(
            x=X_trans, y=Y, callbacks=callbacks,
            validation_split=validation_split,
            batch_size=self.batch_size,
            verbose=verbose, **kwargs)
        self.logger.info("Fitting completed")

    def predict_scores(self, X, **kwargs):
        self.logger.info("Predicting scores")
        n_instances, n_objects, n_features = tensorify(X).get_shape().as_list()
        Y = []
        for i in range(n_instances):
            Y.append(np.arange(n_objects))
        Y = np.array(Y)
        X_trans = self.one_hot_encoder_lr_data_conversion(X, Y)
        return self.model.predict(X_trans, **kwargs)

    def predict(self, X, **kwargs):
        self.logger.info("Predicting ranks")
        return LabelRanker.predict(self, X, **kwargs)
