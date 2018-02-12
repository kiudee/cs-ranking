import logging
from collections import OrderedDict

import numpy as np
from keras import Input, backend as K, optimizers
from keras.engine import Model
from keras.layers import Dense, Lambda, add
from keras.losses import binary_crossentropy
from keras.metrics import top_k_categorical_accuracy, binary_accuracy
from keras.regularizers import l2
from sklearn.utils import check_random_state

from csrank.callbacks import EarlyStoppingWithWeights
from csrank.constants import REGULARIZATION_FACTOR, LEARNING_RATE, BATCH_SIZE, \
    LR_DEFAULT_RANGE, REGULARIZATION_FACTOR_DEFAULT_RANGE, \
    BATCH_SIZE_DEFAULT_RANGE, EARLY_STOPPING_PATIENCE, EARLY_STOPPING_PATIENCE_DEFAULT_RANGE
from csrank.layers import NormalizedDense
from csrank.objectranking.constants import THRESHOLD, N_HIDDEN_LAYERS, N_HIDDEN_UNITS, \
    N_HIDDEN_LAYERS_DEFAULT_RANGES, \
    N_UNITS_DEFAULT_RANGES
from csrank.objectranking.object_ranker import ObjectRanker
from csrank.tunable import Tunable
from csrank.util import tunable_parameters_ranges
from ..dataset_reader.objectranking.util import generate_complete_pairwise_dataset

__all__ = ['RankNet']


class RankNet(ObjectRanker, Tunable):
    _tunable = None
    _use_early_stopping = None

    def __init__(self, n_features, n_hidden=2, n_units=8,
                 loss_function=binary_crossentropy, batch_normalization=True,
                 kernel_regularizer=l2(l=0.01), non_linearities='relu',
                 optimizer="adam", metrics=[top_k_categorical_accuracy, binary_accuracy],
                 use_early_stopping=False, es_patience=300, batch_size=256, random_state=None, **kwargs):

        self.logger = logging.getLogger("RankNet")
        self.n_features = n_features
        self.batch_normalization = batch_normalization
        self.non_linearities = non_linearities
        self.early_stopping = EarlyStoppingWithWeights(patience=es_patience)
        self._use_early_stopping = use_early_stopping
        self.metrics = metrics
        self.kernel_regularizer = kernel_regularizer
        self.loss_function = loss_function
        self.optimizer = optimizers.get(optimizer)
        self._construct_layers(n_hidden, n_units)
        self.threshold_instances = THRESHOLD
        self.batch_size = batch_size
        self.random_state = check_random_state(random_state)

    def _construct_layers(self, n_hidden=2, n_units=8, **kwargs):
        self.x1 = Input(shape=(self.n_features,))
        self.x2 = Input(shape=(self.n_features,))
        self.output_node = Dense(1, activation='sigmoid',
                                 kernel_regularizer=self.kernel_regularizer)
        self.output_layer_score = Dense(1, activation='linear')
        if self.batch_normalization:
            self.hidden_layers = [NormalizedDense(n_units, name="hidden_{}".format(x),
                                                  kernel_regularizer=self.kernel_regularizer,
                                                  activation=self.non_linearities
                                                  ) for x in range(n_hidden)]
        else:
            self.hidden_layers = [Dense(n_units, name="hidden_{}".format(x),
                                        kernel_regularizer=self.kernel_regularizer,
                                        activation=self.non_linearities)
                                  for x in range(n_hidden)]
        assert len(self.hidden_layers) == n_hidden

    def fit(self, X, Y, epochs=10, log_callbacks=None,
            validation_split=0.1, verbose=0, **kwd):

        self.logger.debug('Creating the Dataset')

        garbage, X1, X2, garbage, Y_single = generate_complete_pairwise_dataset(X, Y)
        del garbage
        if (X1.shape[0] > self.threshold_instances):
            indicies = self.random_state.choice(X1.shape[0], self.threshold_instances, replace=False)
            X1 = X1[indicies, :]
            X2 = X2[indicies, :]
            Y_single = Y_single[indicies]

        self.logger.debug('Finished the Dataset')

        self.logger.debug('Creating the model')

        output = self.construct_model()

        callbacks = []
        if log_callbacks is None:
            log_callbacks = []
        callbacks.extend(log_callbacks)
        callbacks = self.set_init_lr_callback(callbacks)

        if self._use_early_stopping:
            callbacks.append(self.early_stopping)

        self.logger.info("Callbacks {}".format(', '.join([c.__name__ for c in callbacks])))
        # Model with input as two objects and output as probability of x1>x2
        self.model = Model(inputs=[self.x1, self.x2], outputs=output)

        self.model.compile(loss=self.loss_function, optimizer=self.optimizer, metrics=self.metrics)
        self.logger.debug('Finished Creating the model, now fitting started')

        self.model.fit([X1, X2], Y_single, batch_size=self.batch_size, epochs=epochs,
                       callbacks=callbacks, validation_split=validation_split,
                       verbose=verbose, **kwd)
        self.scoring_model = self._create_scoring_model()

        self.logger.debug('Fitting Complete')

    def construct_model(self):
        # weight sharing using same hidden layer for two objects
        enc_x1 = self.hidden_layers[0](self.x1)
        enc_x2 = self.hidden_layers[0](self.x2)
        neg_x2 = Lambda(lambda x: -x)(enc_x2)
        for hidden_layer in self.hidden_layers[1:]:
            enc_x1 = hidden_layer(enc_x1)
            neg_x2 = hidden_layer(neg_x2)
        merged_inputs = add([enc_x1, neg_x2])
        output = self.output_node(merged_inputs)
        return output

    def _create_scoring_model(self):
        inp = Input(shape=(self.n_features,))
        x = inp
        for hidden_layer in self.hidden_layers:
            x = hidden_layer(x)
        output_score = self.output_node(x)
        model = Model(inputs=[inp], outputs=output_score)
        return model

    def predict(self, X, **kwargs):
        return super().predict(X, **kwargs)

    def _predict_scores_fixed(self, X, **kwargs):
        # assert X1.shape[1] == self.n_features
        n_instances, n_objects, n_features = X.shape
        self.logger.info("For Test instances {} objects {} features {}".format(n_instances, n_objects, n_features))
        scores = []
        for X1 in X:
            score = self.scoring_model.predict(X1, **kwargs)
            score = score.flatten()
            scores.append(score)
        scores = np.array(scores)
        self.logger.info("Done predicting scores")
        return scores

    def predict_scores(self, X, **kwargs):
        return super().predict_scores(X, **kwargs)

    def predict_pair(self, X1, X2, **kwargs):
        pairwise = np.empty(2)
        pairwise[0] = self.model.predict([X1, X2], **kwargs)
        pairwise[1] = self.model.predict([X2, X1], **kwargs)
        return pairwise

    def evaluate(self, X1_test, X2_test, Y_test, **kwargs):
        return self.model.evaluate([X1_test, X2_test], Y_test, **kwargs)

    @classmethod
    def set_tunable_parameter_ranges(cls, param_ranges_dict):
        logger = logging.getLogger('RankNet')
        return tunable_parameters_ranges(cls, logger, param_ranges_dict)

    def set_tunable_parameters(self, point):
        named = Tunable.set_tunable_parameters(self, point)
        hidden_layers_created = False
        for name, param in named.items():
            if name in [N_HIDDEN_LAYERS, N_HIDDEN_UNITS, REGULARIZATION_FACTOR]:
                self.kernel_regularizer = l2(l=named[REGULARIZATION_FACTOR])
                if not hidden_layers_created:
                    self._construct_layers(**named)
                hidden_layers_created = True
            elif name == LEARNING_RATE:
                K.set_value(self.optimizer.lr, param)
            elif name == EARLY_STOPPING_PATIENCE:
                self.early_stopping.patience = param
            elif name == BATCH_SIZE:
                self.batch_size = param
            else:
                self.logger.warning(
                    'This ranking algorithm does not support a tunable parameter called {}'.format(name))

    @classmethod
    def tunable_parameters(cls):
        if cls._tunable is None:
            cls._tunable = OrderedDict([
                (N_HIDDEN_LAYERS, N_HIDDEN_LAYERS_DEFAULT_RANGES),
                (N_HIDDEN_UNITS, N_UNITS_DEFAULT_RANGES),
                (LEARNING_RATE, LR_DEFAULT_RANGE),
                (REGULARIZATION_FACTOR, REGULARIZATION_FACTOR_DEFAULT_RANGE),
                (BATCH_SIZE, BATCH_SIZE_DEFAULT_RANGE),
            ])
            if cls._use_early_stopping:
                cls._tunable[EARLY_STOPPING_PATIENCE] = EARLY_STOPPING_PATIENCE_DEFAULT_RANGE
