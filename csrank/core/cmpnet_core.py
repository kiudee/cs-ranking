import logging
from itertools import permutations

import numpy as np
import tensorflow as tf
from keras import optimizers, Input, Model, backend as K
from keras.layers import Dense, concatenate
from keras.optimizers import SGD
from keras.regularizers import l2
from sklearn.utils import check_random_state

from csrank.constants import allowed_dense_kwargs
from csrank.layers import NormalizedDense
from csrank.learner import Learner
from csrank.util import print_dictionary


class CmpNetCore(Learner):
    def __init__(self, n_object_features, n_hidden=2, n_units=8, loss_function='binary_crossentropy',
                 batch_normalization=True, kernel_regularizer=l2(l=1e-4), kernel_initializer='lecun_normal',
                 activation='relu', optimizer=SGD(lr=1e-4, nesterov=True, momentum=0.9), metrics=['binary_accuracy'],
                 batch_size=256, random_state=None, **kwargs):
        self.logger = logging.getLogger("CmpNet")
        self.n_object_features = n_object_features
        self.batch_normalization = batch_normalization
        self.activation = activation
        self.hash_file = None

        self.batch_size = batch_size

        self.metrics = metrics
        self.kernel_regularizer = kernel_regularizer
        self.kernel_initializer = kernel_initializer
        self.loss_function = loss_function

        self.optimizer = optimizers.get(optimizer)
        self._optimizer_config = self.optimizer.get_config()

        self.n_hidden = n_hidden
        self.n_units = n_units
        keys = list(kwargs.keys())
        for key in keys:
            if key not in allowed_dense_kwargs:
                del kwargs[key]
        self.kwargs = kwargs
        self.threshold_instances = int(1e10)
        self.random_state = check_random_state(random_state)
        self.model = None
        self._construct_layers(kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.kernel_initializer,
                               activation=self.activation, **self.kwargs)

    def _construct_layers(self, **kwargs):

        self.output_node = Dense(1, activation='sigmoid', kernel_regularizer=self.kernel_regularizer)
        self.x1 = Input(shape=(self.n_object_features,))
        self.x2 = Input(shape=(self.n_object_features,))
        if self.batch_normalization:
            self.hidden_layers = [NormalizedDense(self.n_units, name="hidden_{}".format(x), **kwargs) for x in
                                  range(self.n_hidden)]
        else:
            self.hidden_layers = [Dense(self.n_units, name="hidden_{}".format(x), **kwargs) for x in
                                  range(self.n_hidden)]
        assert len(self.hidden_layers) == self.n_hidden

    def _convert_instances_(self, X, Y):
        raise NotImplemented

    def construct_model(self):
        """
            Construct the CmpNet which is used to approximate the :math:`U_1(x_i,x_j)`. For each pair of objects in
            :math:`x_i, x_j \in Q` we construct two sub-networks with weight sharing in all hidden layers.
            The output of these networks are connected to two sigmoid units that produces the outputs of the network,
            i.e., :math:`U(x_1,x_2), U(x_2,x_1)` for each pair of objects are evaluated. :math:`U(x_1,x_2)` is a measure
            of how favorable it is to choose :math:`x_1` over :math:`x_2`.

            Returns
            -------
            model: keras :class:`Model`
                Neural network to learn the CmpNet utility score
        """
        x1x2 = concatenate([self.x1, self.x2])
        x2x1 = concatenate([self.x2, self.x1])
        self.logger.debug('Creating the model')
        for hidden in self.hidden_layers:
            x1x2 = hidden(x1x2)
            x2x1 = hidden(x2x1)
        merged_left = concatenate([x1x2, x2x1])
        merged_right = concatenate([x2x1, x1x2])
        N_g = self.output_node(merged_left)
        N_l = self.output_node(merged_right)
        merged_output = concatenate([N_g, N_l])
        model = Model(inputs=[self.x1, self.x2], outputs=merged_output)
        model.compile(loss=self.loss_function, optimizer=self.optimizer, metrics=self.metrics)
        return model

    def fit(self, X, Y, epochs=10, callbacks=None, validation_split=0.1, verbose=0, **kwd):
        """
            Fit a generic preference learning CmptNet on the provided set of queries X and preferences Y of those
            objects. The provided queries and corresponding preferences are of a fixed size (numpy arrays).
            For learning this network the binary cross entropy loss function for a pair of objects
            :math:`x_i, x_j \in Q` is defined as:

            .. math::

                C_{ij} =  -\\tilde{P_{ij}}(0)\\cdot \log(U(x_i,x_j)) - \\tilde{P_{ij}}(1) \\cdot \log(U(x_j,x_i)) \ ,

            where :math:`\\tilde{P_{ij}}` is ground truth probability of the preference of :math:`x_i` over :math:`x_j`.
            :math:`\\tilde{P_{ij}} = (1,0)` if :math:`x_i \succ x_j` else :math:`\\tilde{P_{ij}} = (0,1)`.

            Parameters
            ----------
            X : numpy array
                (n_instances, n_objects, n_features)
                Feature vectors of the objects
            Y : numpy array
                (n_instances, n_objects)
                Preferences in form of Orderings or Choices for given n_objects
            epochs : int
                Number of epochs to run if training for a fixed query size
            callbacks : list
                List of callbacks to be called during optimization
            validation_split : float (range : [0,1])
                Percentage of instances to split off to validate on
            verbose : bool
                Print verbose information
            **kwd :
                Keyword arguments for the fit function
        """
        x1, x2, y_double = self._convert_instances_(X, Y)

        self.logger.debug("Instances created {}".format(x1.shape[0]))
        self.model = self.construct_model()

        self.logger.debug('Finished Creating the model, now fitting started')
        self.model.fit([x1, x2], y_double, batch_size=self.batch_size, epochs=epochs, callbacks=callbacks,
                       validation_split=validation_split, verbose=verbose, **kwd)
        self.logger.debug('Fitting Complete')

    def predict_pair(self, a, b, **kwargs):
        return self.model.predict([a, b], **kwargs)

    def _predict_scores_fixed(self, X, **kwargs):
        n_instances, n_objects, n_features = X.shape
        self.logger.info("Test Set instances {} objects {} features {}".format(*X.shape))
        n2 = n_objects * (n_objects - 1)
        pairs = np.empty((n2, 2, n_features))
        scores = np.empty((n_instances, n_objects))
        for n in range(n_instances):
            for k, (i, j) in enumerate(permutations(range(n_objects), 2)):
                pairs[k] = (X[n, i], X[n, j])
            result = self.predict_pair(pairs[:, 0], pairs[:, 1], **kwargs)[:, 0]
            scores[n] = result.reshape(n_objects, n_objects - 1).mean(axis=1)
            del result
        del pairs
        self.logger.info("Done predicting scores")

        return scores

    def clear_memory(self, **kwargs):
        """
            Clear the memory, restores the currently fitted model back to prevent memory leaks.

            Parameters
            ----------
            **kwargs :
                Keyword arguments for the function
        """
        if self.hash_file is not None:
            self.model.save_weights(self.hash_file)
            K.clear_session()
            sess = tf.Session()
            K.set_session(sess)

            self.optimizer = self.optimizer.from_config(self._optimizer_config)
            self._construct_layers(kernel_regularizer=self.kernel_regularizer,
                                   kernel_initializer=self.kernel_initializer,
                                   activation=self.activation, **self.kwargs)
            self.model = self.construct_model()
            self.model.load_weights(self.hash_file)
        else:
            self.logger.info("Cannot clear the memory")

    def set_tunable_parameters(self, n_hidden=32, n_units=2, reg_strength=1e-4, learning_rate=1e-3, batch_size=128,
                               **point):
        """
            Set tunable parameters of the CmpNet network to the values provided.

            Parameters
            ----------
            n_hidden: int
                Number of hidden layers used in the scoring network
            n_units: int
                Number of hidden units in each layer of the scoring network
            reg_strength: float
                Regularization strength of the regularizer function applied to the `kernel` weights matrix
            learning_rate: float
                Learning rate of the stochastic gradient descent algorithm used by the network
            batch_size: int
                Batch size to use during training
            point: dict
                Dictionary containing parameter values which are not tuned for the network
        """
        self.n_hidden = n_hidden
        self.n_units = n_units
        self.kernel_regularizer = l2(reg_strength)
        self.batch_size = batch_size
        self.optimizer = self.optimizer.from_config(self._optimizer_config)
        K.set_value(self.optimizer.lr, learning_rate)
        self._construct_layers(kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.kernel_initializer,
                               activation=self.activation, **self.kwargs)
        if len(point) > 0:
            self.logger.warning('This ranking algorithm does not support'
                                ' tunable parameters called: {}'.format(print_dictionary(point)))
