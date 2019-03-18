import logging

import tensorflow as tf
from keras import optimizers, Input, Model, backend as K
from keras.layers import Dense, Lambda, add
from keras.optimizers import SGD
from keras.regularizers import l2
from sklearn.utils import check_random_state

from csrank.constants import allowed_dense_kwargs
from csrank.layers import NormalizedDense
from csrank.learner import Learner
from csrank.util import print_dictionary


class RankNetCore(Learner):
    def __init__(self, n_object_features, n_hidden=2, n_units=8, loss_function='binary_crossentropy',
                 batch_normalization=True, kernel_regularizer=l2(l=1e-4), kernel_initializer='lecun_normal',
                 activation='relu', optimizer=SGD(lr=1e-4, nesterov=True, momentum=0.9), metrics=['binary_accuracy'],
                 batch_size=256, random_state=None, **kwargs):
        """Create an instance of the RankNet architecture.

        RankNet breaks the preferences into pairwise comparisons and learns a
        latent utility model for the objects.

        Parameters
        ----------
        n_object_features : int
            Number of features of the object space
        n_hidden : int
            Number of hidden layers used in the scoring network
        n_units : int
            Number of hidden units in each layer of the scoring network
        loss_function : function or string
            Loss function to be used for the binary decision task of the
            pairwise comparisons
        batch_normalization : bool
            Whether to use batch normalization in each hidden layer
        kernel_regularizer : function
            Regularizer function applied to all the hidden weight matrices.
        kernel_initializer : function or string
                Initialization function for the weights of each hidden layer
        activation : function or string
            Type of activation function to use in each hidden layer
        optimizer : function or string
            Optimizer to use during stochastic gradient descent
        metrics : list
            List of metrics to evaluate during training (can be non-differentiable)
        batch_size : int
            Batch size to use during training
        random_state : int, RandomState instance or None
            Seed of the pseudo-random generator or a RandomState instance
        **kwargs
            Keyword arguments for the algorithms

        References
        ----------
        .. [1] Burges, C. et al. (2005, August).
               "Learning to rank using gradient descent.",
               In Proceedings of the 22nd international conference on Machine
               learning (pp. 89-96). ACM.
        .. [2] Burges, C. J. (2010).
               "From ranknet to lambdarank to lambdamart: An overview.",
               Learning, 11(23-581), 81.
        """
        self.logger = logging.getLogger(RankNetCore.__name__)
        self.n_object_features = n_object_features
        self.batch_normalization = batch_normalization
        self.activation = activation
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
        self.batch_size = batch_size
        self._scoring_model = None
        self.model = None
        self.hash_file = None
        self.random_state = check_random_state(random_state)
        self._construct_layers(kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.kernel_initializer,
                               activation=self.activation, **self.kwargs)

    def _construct_layers(self, **kwargs):
        self.logger.info("n_hidden {}, n_units {}".format(self.n_hidden, self.n_units))
        self.x1 = Input(shape=(self.n_object_features,))
        self.x2 = Input(shape=(self.n_object_features,))
        self.output_node = Dense(1, activation='sigmoid', kernel_regularizer=self.kernel_regularizer)
        self.output_layer_score = Dense(1, activation='linear')
        if self.batch_normalization:
            self.hidden_layers = [NormalizedDense(self.n_units, name="hidden_{}".format(x), **kwargs) for x in
                                  range(self.n_hidden)]
        else:
            self.hidden_layers = [Dense(self.n_units, name="hidden_{}".format(x), **kwargs) for x in
                                  range(self.n_hidden)]
        assert len(self.hidden_layers) == self.n_hidden

    def fit(self, X, Y, epochs=10, callbacks=None, validation_split=0.1, verbose=0, **kwd):

        X1, X2, Y_single = self.convert_instances(X, Y)

        self.logger.debug("Instances created {}".format(X1.shape[0]))
        self.logger.debug('Creating the model')

        output = self.construct_model()

        # Model with input as two objects and output as probability of x1>x2
        self.model = Model(inputs=[self.x1, self.x2], outputs=output)

        self.model.compile(loss=self.loss_function, optimizer=self.optimizer, metrics=self.metrics)
        self.logger.debug('Finished Creating the model, now fitting started')

        self.model.fit([X1, X2], Y_single, batch_size=self.batch_size, epochs=epochs, callbacks=callbacks,
                       validation_split=validation_split, verbose=verbose, **kwd)

        self.logger.debug('Fitting Complete')

    @property
    def scoring_model(self):
        if self._scoring_model is None:
            self.logger.info('creating scoring model')
            inp = Input(shape=(self.n_object_features,))
            x = inp
            for hidden_layer in self.hidden_layers:
                x = hidden_layer(x)
            output_score = self.output_node(x)
            self._scoring_model = Model(inputs=[inp], outputs=output_score)
        return self._scoring_model

    def convert_instances(self, X, Y):
        raise NotImplemented

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

    def _predict_scores_fixed(self, X, **kwargs):
        n_instances, n_objects, n_features = X.shape
        self.logger.info("Test Set instances {} objects {} features {}".format(*X.shape))
        X1 = X.reshape(n_instances * n_objects, n_features)
        scores = self.scoring_model.predict(X1, **kwargs)
        scores = scores.reshape(n_instances, n_objects)
        self.logger.info("Done predicting scores")
        return scores

    def clear_memory(self, **kwargs):
        if self.hash_file is not None:
            self.model.save_weights(self.hash_file)
            K.clear_session()
            sess = tf.Session()
            K.set_session(sess)

            self._scoring_model = None
            self.optimizer = self.optimizer.from_config(self._optimizer_config)
            self._construct_layers(kernel_regularizer=self.kernel_regularizer,
                                   kernel_initializer=self.kernel_initializer,
                                   activation=self.activation, **self.kwargs)
            output = self.construct_model()
            self.model = Model(inputs=[self.x1, self.x2], outputs=output)
            self.model.load_weights(self.hash_file)
        else:
            self.logger.info("Cannot clear the memory")

    def set_tunable_parameters(self, n_hidden=32, n_units=2, reg_strength=1e-4, learning_rate=1e-3, batch_size=128,
                               **point):
        self.n_hidden = n_hidden
        self.n_units = n_units
        self.kernel_regularizer = l2(reg_strength)
        self.batch_size = batch_size
        self.optimizer = self.optimizer.from_config(self._optimizer_config)
        K.set_value(self.optimizer.lr, learning_rate)
        self._scoring_model = None
        self._construct_layers(kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.kernel_initializer,
                               activation=self.activation, **self.kwargs)
        if len(point) > 0:
            self.logger.warning('This ranking algorithm does not support'
                                ' tunable parameters called: {}'.format(print_dictionary(point)))
