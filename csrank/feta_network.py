import logging
from itertools import permutations, combinations

import numpy as np
import tensorflow as tf
from keras import optimizers, Input, Model, backend as K
from keras.layers import Dense, concatenate, Lambda, add
from keras.regularizers import l2
from sklearn.utils import check_random_state

from csrank.constants import allowed_dense_kwargs
from csrank.layers import NormalizedDense
from csrank.learner import Learner
from csrank.losses import hinged_rank_loss
from csrank.util import tensorify, print_dictionary


class FETANetwork(Learner):
    def __init__(self, n_objects, n_object_features, hash_file, n_hidden=2, n_units=8,
                 add_zeroth_order_model=False, max_number_of_objects=5,
                 num_subsample=5, loss_function=hinged_rank_loss, batch_normalization=False,
                 kernel_regularizer=l2(l=1e-4), kernel_initializer='lecun_normal', activation='selu',
                 optimizer="adam", metrics=None, batch_size=256, random_state=None, **kwargs):
        """
        Create a FETA-network architecture for object ranking.
        Training and prediction complexity is quadratic in the number of objects.

        Parameters
        ----------
        n_objects : int
            Number of objects to be ranked
        n_object_features : int
            Dimensionality of the feature space of each object
        hash_file: str
            File path of the model where the weights are stored to get the predictions after clearing the memory
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
        kernel_initializer : function or string
            Initialization function for the weights of each hidden layer
        activation : string or function
            Activation function to use in the hidden units
        optimizer : string or function
            Stochastic gradient optimizer
        metrics : list
            List of evaluation metrics (can be non-differentiable)
        batch_size : int
            Batch size to use for training
        random_state : int or object
            Numpy random state
        **kwargs
            Keyword arguments for the hidden units
        """
        self.logger = logging.getLogger(FETANetwork.__name__)

        self.random_state = check_random_state(random_state)

        self.kernel_regularizer = kernel_regularizer
        self.kernel_initializer = kernel_initializer
        self.batch_normalization = batch_normalization
        self.activation = activation
        self.loss_function = loss_function
        self.metrics = metrics
        self._n_objects = n_objects
        self.max_number_of_objects = max_number_of_objects
        self.num_subsample = num_subsample
        self.n_object_features = n_object_features
        self.batch_size = batch_size
        self.hash_file = hash_file
        self.optimizer = optimizers.get(optimizer)
        self._optimizer_config = self.optimizer.get_config()
        self._use_zeroth_model = add_zeroth_order_model
        self.n_hidden = n_hidden
        self.n_units = n_units
        keys = list(kwargs.keys())
        for key in keys:
            if key not in allowed_dense_kwargs:
                del kwargs[key]
        self.kwargs = kwargs

        self._construct_layers(kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.kernel_initializer,
                               activation=self.activation, **self.kwargs)
        self._pairwise_model = None
        self.model = None
        self.zero_order_model = None

    @property
    def n_objects(self):
        if self._n_objects > self.max_number_of_objects:
            return self.max_number_of_objects
        return self._n_objects

    def _construct_layers(self, **kwargs):
        self.input_layer = Input(shape=(self.n_objects, self.n_object_features))
        # Todo: Variable sized input
        # X = Input(shape=(None, n_features))
        if self.batch_normalization:
            if self._use_zeroth_model:
                self.hidden_layers_zeroth = [NormalizedDense(self.n_units, name="hidden_zeroth_{}".format(x), **kwargs)
                                             for x in range(self.n_hidden)]
            self.hidden_layers = [NormalizedDense(self.n_units, name="hidden_{}".format(x), **kwargs) for x in
                                  range(self.n_hidden)]
        else:
            if self._use_zeroth_model:
                self.hidden_layers_zeroth = [Dense(self.n_units, name="hidden_zeroth_{}".format(x), **kwargs) for x in
                                             range(self.n_hidden)]
            self.hidden_layers = [Dense(self.n_units, name="hidden_{}".format(x), **kwargs) for x in
                                  range(self.n_hidden)]
        assert len(self.hidden_layers) == self.n_hidden
        self.output_node = Dense(1, activation="sigmoid", kernel_regularizer=self.kernel_regularizer)
        if self._use_zeroth_model:
            self.output_node_zeroth = Dense(1, activation="sigmoid", kernel_regularizer=self.kernel_regularizer)

    def _create_zeroth_order_model(self):
        inp = Input(shape=(self.n_object_features,))

        x = inp
        for hidden in self.hidden_layers_zeroth:
            x = hidden(x)
        zeroth_output = self.output_node_zeroth(x)

        return Model(inputs=[inp], outputs=zeroth_output)

    @property
    def pairwise_model(self):
        if self._pairwise_model is None:
            self.logger.info('Creating pairwise model')
            x1 = Input(shape=(self.n_object_features,))
            x2 = Input(shape=(self.n_object_features,))

            x1x2 = concatenate([x1, x2])
            x2x1 = concatenate([x2, x1])

            for hidden in self.hidden_layers:
                x1x2 = hidden(x1x2)
                x2x1 = hidden(x2x1)

            merged_left = concatenate([x1x2, x2x1])
            merged_right = concatenate([x2x1, x1x2])

            n_g = self.output_node(merged_left)
            n_l = self.output_node(merged_right)

            merged_output = concatenate([n_g, n_l])
            self._pairwise_model = Model(inputs=[x1, x2], outputs=merged_output)
        return self._pairwise_model

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
            scores[n] = 1. / (1. + np.exp(-scores[n]))
            del result
        del pairs
        return scores

    def fit(self, X, Y, epochs=10, callbacks=None,
            validation_split=0.1, verbose=0, **kwd):
        self.logger.debug('Enter fit function...')

        X, Y = self.sub_sampling(X, Y)
        scores = self.construct_model()
        self.model = Model(inputs=self.input_layer, outputs=scores)

        if self._use_zeroth_model:
            self.zero_order_model = self._create_zeroth_order_model()

        self.logger.debug('Compiling complete model...')
        self.model.compile(loss=self.loss_function, optimizer=self.optimizer, metrics=self.metrics)
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

            n_g = self.output_node(merged_left)
            n_l = self.output_node(merged_right)

            outputs[i].append(n_g)
            outputs[j].append(n_l)
        # convert rows of pairwise matrix to keras layers:
        outputs = [concatenate(x) for x in outputs]
        # compute utility scores:
        sum_func = lambda s: K.mean(s, axis=1, keepdims=True)
        scores = [Lambda(sum_func)(x) for x in outputs]
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
            idx += np.arange(start=0, stop=self._n_objects, step=bucket_size)[
                   :self.n_objects]
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

    def set_tunable_parameters(self, n_hidden=32,
                               n_units=2,
                               reg_strength=1e-4,
                               learning_rate=1e-3,
                               batch_size=128, **point):
        self.n_hidden = n_hidden
        self.n_units = n_units
        self.kernel_regularizer = l2(reg_strength)
        self.batch_size = batch_size
        self.optimizer = self.optimizer.from_config(self._optimizer_config)
        K.set_value(self.optimizer.lr, learning_rate)
        self._pairwise_model = None
        self._construct_layers(kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.kernel_initializer,
                               activation=self.activation, **self.kwargs)
        if len(point) > 0:
            self.logger.warning('This ranking algorithm does not support tunable parameters'
                                ' called: {}'.format(print_dictionary(point)))

    def clear_memory(self, n_objects, **kwargs):
        self.model.save_weights(self.hash_file)
        K.clear_session()
        sess = tf.Session()
        K.set_session(sess)

        self._pairwise_model = None
        self.optimizer = self.optimizer.from_config(self._optimizer_config)
        self._construct_layers(kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.kernel_initializer,
                               activation=self.activation, **self.kwargs)
        scores = self.construct_model()
        self.model = Model(inputs=self.input_layer, outputs=scores)
        self.model.load_weights(self.hash_file)
