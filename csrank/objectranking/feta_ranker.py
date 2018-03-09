import logging
import operator
from collections import OrderedDict
from itertools import combinations, permutations

import numpy as np
from keras import Input, backend as K, optimizers
from keras.engine import Model
from keras.layers import Dense, concatenate, Lambda, add
from keras.regularizers import l2
from sklearn.utils import check_random_state

from csrank.callbacks import EarlyStoppingWithWeights
from csrank.constants import LEARNING_RATE, LR_DEFAULT_RANGE, BATCH_SIZE, \
    REGULARIZATION_FACTOR, BATCH_SIZE_DEFAULT_RANGE, REGULARIZATION_FACTOR_DEFAULT_RANGE, EARLY_STOPPING_PATIENCE, \
    EARLY_STOPPING_PATIENCE_DEFAULT_RANGE
from csrank.layers import NormalizedDense
from csrank.losses import hinged_rank_loss
from csrank.objectranking.constants import N_UNITS_DEFAULT_RANGES, N_HIDDEN_LAYERS_DEFAULT_RANGES, N_HIDDEN_UNITS, \
    N_HIDDEN_LAYERS
from csrank.objectranking.object_ranker import ObjectRanker
from csrank.tunable import Tunable
from csrank.util import tunable_parameters_ranges, tensorify

__all__ = ['FETANetwork']


class FETANetwork(ObjectRanker, Tunable):
    """
        Create a FETA-network architecture for object ranking.
        Training and prediction complexity is quadratic in the number of objects.

        Parameters
        ----------
        n_objects : int
            Number of objects to be ranked
        n_features : int
            Dimensionality of the feature space of each object
        n_hidden : int
            Number of hidden layers
        n_units : int
            Number of hidden units in each layer
        add_zeroth_order_model : bool
            True if the model should include a latent utility function
        max_number_of_objects : int
            The maximum number of objects to train from
        num_subsample : int
            Number of objects to subsample to
        loss_function : function
            Differentiable loss function for the score vector
        batch_normalization : bool
            Whether to use batch normalization in the hidden layers
        kernel_regularizer : function
            Regularizer to use in the hidden units
        non_linearities : string or function
            Activation function to use in the hidden units
        optimizer : string or function
            Stochastic gradient optimizer
        metrics : list
            List of evaluation metrics (can be non-differentiable)
        use_early_stopping : bool
            Whether to use early stopping during training
        es_patience : int
            Number of iterations to wait for no improvement in early stopping
        batch_size : int
            Batch size to use for training
        random_state : int or object
            Numpy random state
        **kwargs
            Keyword arguments for the hidden units
    """
    _tunable = None
    _use_early_stopping = None

    def __init__(self, n_objects, n_features, n_hidden=2, n_units=8,
                 add_zeroth_order_model=False, max_number_of_objects=5,
                 num_subsample=5,
                 loss_function=hinged_rank_loss, batch_normalization=False,
                 kernel_regularizer=l2(l=0.01), non_linearities='selu',
                 optimizer="adam", metrics=None, use_early_stopping=False,
                 es_patience=300, batch_size=256, random_state=None, **kwargs):
        self.logger = logging.getLogger("FETANetwork")

        self.random_state = check_random_state(random_state)

        self.kernel_regularizer = kernel_regularizer
        self.batch_normalization = batch_normalization
        self.non_linearities = non_linearities
        self.loss_function = loss_function
        self.metrics = metrics
        self._n_objects = n_objects
        self.max_number_of_objects = max_number_of_objects
        self.num_subsample = num_subsample
        self.n_features = n_features
        self._use_early_stopping = use_early_stopping
        self.batch_size = batch_size

        self.optimizer = optimizers.get(optimizer)
        self._use_zeroth_model = add_zeroth_order_model
        self.early_stopping = EarlyStoppingWithWeights(patience=es_patience)
        self._construct_layers(n_hidden, n_units)

    @property
    def n_objects(self):
        if self._n_objects > self.max_number_of_objects:
            return self.max_number_of_objects
        return self._n_objects

    def _construct_layers(self, n_hidden=2, n_units=16, **kwargs):
        self.input_layer = Input(shape=(self.n_objects, self.n_features))
        # Todo: Variable sized input
        # X = Input(shape=(None, n_features))
        if self.batch_normalization:
            if self._use_zeroth_model:
                self.hidden_layers_zeroth = [
                    NormalizedDense(n_units, name="hidden_zeroth_{}".format(x),
                                    kernel_regularizer=self.kernel_regularizer,
                                    activation=self.non_linearities
                                    )
                    for x in range(n_hidden)
                ]
            self.hidden_layers = [
                NormalizedDense(n_units, name="hidden_{}".format(x),
                                kernel_regularizer=self.kernel_regularizer,
                                activation=self.non_linearities
                                )
                for x in range(n_hidden)
            ]
        else:
            if self._use_zeroth_model:
                self.hidden_layers_zeroth = [
                    Dense(n_units, name="hidden_zeroth_{}".format(x),
                          kernel_regularizer=self.kernel_regularizer,
                          activation=self.non_linearities)
                    for x in range(n_hidden)
                ]
            self.hidden_layers = [
                Dense(n_units, name="hidden_{}".format(x),
                      kernel_regularizer=self.kernel_regularizer,
                      activation=self.non_linearities)
                for x in range(n_hidden)
            ]
        assert len(self.hidden_layers) == n_hidden
        self.output_node = Dense(1, activation='sigmoid',
                                 kernel_regularizer=self.kernel_regularizer)
        if self._use_zeroth_model:
            self.output_node_zeroth = Dense(1, activation='sigmoid',
                                            kernel_regularizer=self.kernel_regularizer)

    def _create_zeroth_order_model(self):
        inp = Input(shape=(self.n_features,))

        x = inp
        for hidden in self.hidden_layers_zeroth:
            x = hidden(x)
        zeroth_output = self.output_node_zeroth(x)

        return Model(inputs=[inp], outputs=zeroth_output)

    def _create_pairwise_model(self):
        x1 = Input(shape=(self.n_features,))
        x2 = Input(shape=(self.n_features,))

        x1x2 = concatenate([x1, x2])
        x2x1 = concatenate([x2, x1])

        for hidden in self.hidden_layers:
            x1x2 = hidden(x1x2)
            x2x1 = hidden(x2x1)

        merged_left = concatenate([x1x2, x2x1])
        merged_right = concatenate([x2x1, x1x2])

        N_g = self.output_node(merged_left)
        N_l = self.output_node(merged_right)

        merged_output = concatenate([N_g, N_l])

        return Model(inputs=[x1, x2], outputs=merged_output)

    def _predict_pair(self, a, b, only_pairwise=False, **kwargs):
        # TODO: Is this working correctly?
        pairwise = self.pairwise_model.predict([a, b], **kwargs)
        if not only_pairwise and self._use_zeroth_model:
            utility_a = self.zero_order_model.predict([a])
            utility_b = self.zero_order_model.predict([b])
            return pairwise + (utility_a, utility_b)
        return pairwise

    def _predict_scores_using_pairs(self, X, **kwd):
        n_instances, n_objects, n_features = X.shape
        n2 = n_objects * (n_objects - 1)
        pairs = np.empty((n2, 2, n_features))
        scores = np.zeros((n_instances, n_objects))
        for n in range(n_instances):
            if self._use_zeroth_model:
                scores[n] = self.zero_order_model.predict(X[n]).ravel()
            for k, (i, j) in enumerate(permutations(range(n_objects), 2)):
                pairs[k] = (X[n, i], X[n, j])
            result = self._predict_pair(pairs[:, 0], pairs[:, 1],
                                        only_pairwise=True, **kwd)[:, 0]
            scores[n] += result.reshape(n_objects, n_objects - 1).mean(axis=1)
            del result
        del pairs
        return scores

    def fit(self, X, Y, epochs=10, log_callbacks=None,
            validation_split=0.1, verbose=0, **kwd):
        self.logger.debug('Enter fit function...')

        X, Y = self.sub_sampling(X, Y)
        scores = self.construct_model()
        self.model = Model(inputs=self.input_layer, outputs=scores)

        self.pairwise_model = self._create_pairwise_model()
        if self._use_zeroth_model:
            self.zero_order_model = self._create_zeroth_order_model()

        callbacks = []
        if log_callbacks is None:
            log_callbacks = []
        callbacks.extend(log_callbacks)
        callbacks = self.set_init_lr_callback(callbacks)

        if self._use_early_stopping:
            callbacks.append(self.early_stopping)
        self.logger.info("Callbacks {}".format(', '.join([c.__name__ for c in callbacks])))

        self.logger.debug('Compiling complete model...')
        self.model.compile(loss=self.loss_function, optimizer=self.optimizer,
                           metrics=self.metrics)
        self.logger.debug('Starting gradient descent...')
        self.model.fit(x=X, y=Y, batch_size=self.batch_size, epochs=epochs,
                       callbacks=callbacks, validation_split=validation_split,
                       verbose=verbose, **kwd)

    def construct_model(self):
        def create_input_lambda(i):
            return Lambda(lambda x: x[:, i])

        if self._use_zeroth_model:
            self.logger.debug('Create 0th order model')
            zeroth_order_outputs = []
            inputs = []
            for i in range(self.n_objects):
                x = create_input_lambda(i)(self.input_layer)
                inputs.append(x)
                for hidden in self.hidden_layers_zeroth:
                    x = hidden(x)
                zeroth_order_outputs.append(self.output_node_zeroth(x))
            zeroth_order_scores = concatenate(zeroth_order_outputs)
            self.logger.debug('0th order model finished')
        self.logger.debug('Create 1st order model')
        outputs = [list() for _ in range(self.n_objects)]
        for i, j in combinations(range(self.n_objects), 2):
            if self._use_zeroth_model:
                x1 = inputs[i]
                x2 = inputs[j]
            else:
                x1 = create_input_lambda(i)(self.input_layer)
                x2 = create_input_lambda(j)(self.input_layer)
            x1x2 = concatenate([x1, x2])
            x2x1 = concatenate([x2, x1])

            for hidden in self.hidden_layers:
                x1x2 = hidden(x1x2)
                x2x1 = hidden(x2x1)

            merged_left = concatenate([x1x2, x2x1])
            merged_right = concatenate([x2x1, x1x2])

            N_g = self.output_node(merged_left)
            N_l = self.output_node(merged_right)

            outputs[i].append(N_g)
            outputs[j].append(N_l)
        # convert rows of pairwise matrix to keras layers:
        outputs = [concatenate(x) for x in outputs]
        # compute utility scores:
        sum_fun = lambda s: K.mean(s, axis=1, keepdims=True)
        scores = [Lambda(sum_fun)(x) for x in outputs]
        scores = concatenate(scores)
        self.logger.debug('1st order model finished')
        if self._use_zeroth_model:
            scores = add([scores, zeroth_order_scores])
        return scores

    def sub_sampling(self, X, Y):
        if self._n_objects > self.max_number_of_objects:
            bucket_size = int(self._n_objects / self.max_number_of_objects)
            idx = self.random_state.randint(bucket_size,
                                            size=(len(X), self.n_objects))
            # TODO: subsampling multiple rankings
            idx += np.arange(start=0, stop=self._n_objects, step=bucket_size)[:self.n_objects]
            X = X[np.arange(X.shape[0])[:, None], idx]
            Y = Y[np.arange(X.shape[0])[:, None], idx]
            tmp_sort = Y.argsort(axis=-1)
            Y = np.empty_like(Y)
            Y[np.arange(len(X))[:, None], tmp_sort] = np.arange(self.n_objects)
        return X, Y

    def _predict_scores_fixed(self, X, **kwargs):
        n_instances, n_objects, n_features = tensorify(X).get_shape().as_list()
        self.logger.info("For Test instances {} objects {} features {}".format(n_instances, n_objects, n_features))
        if self.max_number_of_objects < self._n_objects or self.n_objects != n_objects:
            scores = self._predict_scores_using_pairs(X, **kwargs)
        else:
            scores = self.model.predict(X, **kwargs)
        self.logger.info("Done predicting scores")
        return scores

    def evaluate(self, X, Y, **kwargs):
        scores = self.model.evaluate(X, Y, **kwargs)
        return scores

    def predict_scores(self, X, **kwargs):
        return super().predict_scores(X, **kwargs)

    def predict(self, X, **kwargs):
        return ObjectRanker.predict(self, X, **kwargs)

    @classmethod
    def set_tunable_parameter_ranges(cls, param_ranges_dict):
        logger = logging.getLogger("FETANetwork")
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
                self.logger.warning('This ranking algorithm does not support'
                                    ' a tunable parameter called {}'.format(name))

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