import logging

import keras.backend as K
import numpy as np
import tensorflow as tf
from keras import optimizers
from keras.layers import Input, Dense
from keras.layers.merge import concatenate
from keras.models import Model
from keras.optimizers import SGD
from keras.regularizers import l2
from sklearn.utils import check_random_state

from csrank.constants import allowed_dense_kwargs
from csrank.layers import DeepSet, create_input_lambda
from csrank.learner import Learner
from csrank.util import print_dictionary

__all__ = ['FATENetwork', 'FATENetworkCore']


class FATENetworkCore(Learner):
    def __init__(self, n_hidden_joint_layers=32, n_hidden_joint_units=32, activation='selu',
                 kernel_initializer='lecun_normal', kernel_regularizer=l2(l=0.01),
                 optimizer=SGD(lr=1e-4, nesterov=True, momentum=0.9), batch_size=256, random_state=None, **kwargs):
        """
            Create a FATE-network architecture.
            Training and prediction complexity is linear in the number of objects.

            Parameters
            ----------
            n_hidden_joint_layers : int
                Number of joint layers.
            n_hidden_joint_units : int
                Number of hidden units in each joint layer
            activation : string or function
                Activation function to use in the hidden units
            kernel_initializer : function or string
                Initialization function for the weights of each hidden layer
            kernel_regularizer : function or string
                Regularizer to use in the hidden units
            optimizer : string or function
                Stochastic gradient optimizer
            batch_size : int
                Batch size to use for training
            random_state : int or object
                Numpy random state
            **kwargs
                Keyword arguments for the hidden units
        """
        self.logger = logging.getLogger(FATENetworkCore.__name__)
        self.random_state = check_random_state(random_state)

        self.n_hidden_joint_layers = n_hidden_joint_layers
        self.n_hidden_joint_units = n_hidden_joint_units

        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        self.batch_size = batch_size
        self.optimizer = optimizers.get(optimizer)
        self._optimizer_config = self.optimizer.get_config()
        self.joint_layers = None
        self.scorer = None
        keys = list(kwargs.keys())
        for key in keys:
            if key not in allowed_dense_kwargs:
                del kwargs[key]
        self.kwargs = kwargs
        self._construct_layers(activation=self.activation, kernel_initializer=self.kernel_initializer,
                               kernel_regularizer=self.kernel_regularizer, **self.kwargs)

    def _construct_layers(self, **kwargs):
        """
            Construct basic layers shared by all ranking algorithms:
             * Joint dense hidden layers
             * Output scoring layer

            Connecting the layers is done in join_input_layers and will be done in implementing classes.

            Parameters
            ----------
            **kwargs
                Keyword arguments passed into the joint layers
        """
        self.logger.info("Construct joint layers hidden units {} and layers {} ".format(self.n_hidden_joint_units,
                                                                                        self.n_hidden_joint_layers))
        # Create joint hidden layers:
        self.joint_layers = []
        for i in range(self.n_hidden_joint_layers):
            self.joint_layers.append(Dense(self.n_hidden_joint_units, name="joint_layer_{}".format(i), **kwargs))

        self.logger.info('Construct output score node')
        self.scorer = Dense(1, name="output_node", activation='linear', kernel_regularizer=self.kernel_regularizer)

    def join_input_layers(self, input_layer, *layers, n_layers, n_objects):
        """
            Accepts input tensors and an arbitrary number of feature tensors and concatenates them into a joint layer.
            The input layers need to be given separately, because they need to be iterated over.

            Parameters
            ----------
            input_layer : input tensor (n_objects, n_features)
            layers : tensors
                A number of tensors representing feature representations
            n_layers : int
                Number of hidden set layers
            n_objects : int
                Number of objects
        """
        self.logger.debug("Joining set representation and joint layers")
        scores = []

        inputs = [create_input_lambda(i)(input_layer) for i in
                  range(n_objects)]

        for i in range(n_objects):
            if n_layers >= 1:
                joint = concatenate([inputs[i], *layers])
            else:
                joint = inputs[i]
            for j in range(self.n_hidden_joint_layers):
                joint = self.joint_layers[j](joint)
            scores.append(self.scorer(joint))
        scores = concatenate(scores, name="final_scores")
        self.logger.debug("Done")

        return scores

    def set_tunable_parameters(self, n_hidden_joint_units=32, n_hidden_joint_layers=2, reg_strength=1e-4,
                               learning_rate=1e-3, batch_size=128, **point):
        self.n_hidden_joint_layers = n_hidden_joint_layers
        self.n_hidden_joint_units = n_hidden_joint_units
        self.kernel_regularizer = l2(reg_strength)
        self.batch_size = batch_size
        # Hack to fix memory leak:
        self.optimizer = self.optimizer.from_config(self._optimizer_config)
        K.set_value(self.optimizer.lr, learning_rate)

        self._construct_layers(activation=self.activation, kernel_initializer=self.kernel_initializer,
                               kernel_regularizer=self.kernel_regularizer, **self.kwargs)

        if len(point) > 0:
            self.logger.warning('This ranking algorithm does not support'
                                ' tunable parameters'
                                ' called: {}'.format(print_dictionary(point)))


class FATENetwork(FATENetworkCore):
    def __init__(self, n_object_features, n_hidden_set_layers=1, n_hidden_set_units=1, **kwargs):
        """
            Create a FATE-network architecture.
            Training and prediction complexity is linear in the number of objects.

            Parameters
            ----------
            n_object_features : int
                Dimensionality of the feature space of each object
            n_hidden_set_layers : int
                Number of hidden set layers.
            n_hidden_set_units : int
                Number of hidden units in each set layer
            **kwargs
                Keyword arguments for the hidden set units
        """
        FATENetworkCore.__init__(self, **kwargs)
        self.logger_gorc = logging.getLogger(FATENetwork.__name__)

        self.n_hidden_set_layers = n_hidden_set_layers
        self.n_hidden_set_units = n_hidden_set_units
        self.n_object_features = n_object_features
        self.model = None
        self.set_layer = None
        self._create_set_layers(activation=self.activation,
                                kernel_initializer=self.kernel_initializer,
                                kernel_regularizer=self.kernel_regularizer)
        self.is_variadic = True
        self.hash_file = None

    def _create_set_layers(self, **kwargs):
        """
            Create layers for learning the representation of the query set. The actual connection of the layers is done
            during fitting, since we do not know the size(s) of the set(s) in advance.
        """
        self.logger_gorc.info("Creating set layers with set units {} set layer {} ".format(self.n_hidden_set_units,
                                                                                           self.n_hidden_set_layers))
        if self.n_hidden_set_layers >= 1:
            self.set_layer = DeepSet(units=self.n_hidden_set_units, layers=self.n_hidden_set_layers, **kwargs)
        else:
            self.set_layer = None

    @staticmethod
    def _bucket_frequencies(X, min_bucket_size=32):
        """
            Calculates the relative frequency of each ranking bucket.

            Parameters
            ----------
            X : dict
                map from n_objects to object queries
            min_bucket_size : int
                Minimum number of instances for a query size to be considered for
                the frequency calculation

            Returns
            -------
            freq : dict
                map from n_objects to frequency in float

        """
        freq = dict()
        total = 0.0
        for n_objects, arr in X.items():
            n_instances = arr.shape[0]
            if n_instances >= min_bucket_size:
                freq[n_objects] = n_instances
                total += freq[n_objects]
            else:
                freq[n_objects] = 0
        for n_objects in freq.keys():
            freq[n_objects] /= total
        return freq

    def _construct_models(self, buckets):
        models = dict()
        n_features = self.n_object_features

        for n_objects in buckets.keys():
            model = self.construct_model(n_features, n_objects)
            models[n_objects] = model
        return models

    def get_weights(self, n_objects=None):
        if self.is_variadic:
            if n_objects is not None:
                weights = self.models_[n_objects].get_weights()
            else:
                weights = self.models_[n_objects].get_weights()
        else:
            weights = self.model.get_weights()
        return weights

    def set_weights(self, weights, n_objects=None):
        if self.is_variadic:
            if n_objects is not None:
                self.models_[n_objects].set_weights(weights)
            else:
                self.models_[0].set_weights(weights)
        else:
            self.model.set_weights(weights)

    def _fit(self, X=None, Y=None, generator=None, epochs=35, inner_epochs=1, callbacks=None, validation_split=0.1,
             verbose=0, global_lr=1.0, global_momentum=0.9, min_bucket_size=500, refit=False, optimizer=None, **kwargs):
        if optimizer is not None:
            self.optimizer = optimizer
        if isinstance(X, dict):
            if generator is not None:
                self.logger.error("Variadic training does not support generators yet.")
                raise NotImplementedError
            self.is_variadic = True
            decay_rate = global_lr / epochs
            learning_rate = global_lr
            freq = self._bucket_frequencies(X, min_bucket_size=min_bucket_size)
            bucket_ids = np.array(tuple(X.keys()))

            #  Create models which need to be trained
            #  Note, that the models share all their weights, the only
            #  difference is the compute graph constructed for back propagation.
            if not hasattr(self, 'models_') or refit:
                self.models_ = self._construct_models(X)

            #  Iterate training
            for epoch in range(epochs):

                self.logger.info("Epoch: {}, Learning rate: {}".format(epoch, learning_rate))

                # In the spirit of mini-batch SGD we also shuffle the buckets
                # each epoch:
                np.random.shuffle(bucket_ids)
                self.curr_bucket_id = bucket_ids[0]

                w_before = np.array(self.get_weights())

                for bucket_id in bucket_ids:
                    self.curr_bucket_id = bucket_id
                    # Skip query sizes with too few instances:
                    if X[bucket_id].shape[0] < min_bucket_size:
                        continue

                    # self.set_weights(start)
                    x = X[bucket_id]
                    y = Y[bucket_id]

                    # Save weight vector for momentum:
                    w_old = w_before
                    w_before = np.array(self.get_weights())
                    self.models_[bucket_id].fit(x=x, y=y, epochs=inner_epochs, batch_size=self.batch_size,
                                                validation_split=validation_split, verbose=verbose, **kwargs)
                    w_after = np.array(self.get_weights())
                    self.set_weights(w_before
                                     + learning_rate * freq[bucket_id]
                                     * (w_after - w_before)
                                     + global_momentum * (w_before - w_old))
                learning_rate /= 1 + decay_rate * epoch
        else:
            self.is_variadic = False

            if self.model is None or refit:
                if generator is not None:
                    X, Y = next(iter(generator))

                n_inst, n_objects, n_features = X.shape

                self.model = self.construct_model(n_features, n_objects)
            self.logger.info("Fitting started")
            if generator is None:
                self.model.fit(x=X, y=Y, callbacks=callbacks, epochs=epochs, validation_split=validation_split,
                               batch_size=self.batch_size, verbose=verbose, **kwargs)
            else:
                self.model.fit_generator(generator=generator, callbacks=callbacks, epochs=epochs, verbose=verbose,
                                         **kwargs)
            self.logger.info("Fitting complete")

    def construct_model(self, n_features, n_objects):
        """
            Construct the FATE-network architecture using the :class:`DeepSet` to learn the context representation
            :math:`\\mu_{C(x)}` for the given query set/context :math:`Q=C(x)`. We construct an input tensor of query
            set :math:`Q` of size (n_objects, n_features),iterate over it for each object and concatenate the
            context-representation feature tensor of size :math:`\\lvert  \\mu_{C(x)} \\lvert` into a joint layers.
            So, for each object we share the weights in the joint network and the output of this network is used to
            learn the generalized latent utility score :math:`U (x, \\mu_{C(x)})` of each object :math:`x \in Q`.

            Parameters
            ----------
            n_features: int
                Features of the objects for which the network is constructed
            n_objects: int
                Size of the query sets for which the network is constructed

            Returns
            -------
             model: keras :class:`Model`
                Neural network to learn the FATE utility score

        """
        input_layer = Input(shape=(n_objects, n_features), name="input_node")
        set_repr = self.set_layer(input_layer)
        scores = self.join_input_layers(input_layer, set_repr, n_objects=n_objects, n_layers=self.n_hidden_set_layers)
        model = Model(inputs=input_layer, outputs=scores)

        model.compile(loss=self.loss_function, optimizer=self.optimizer, metrics=self.metrics)
        return model

    def fit(self, X, Y, epochs=35, inner_epochs=1, callbacks=None, validation_split=0.1, verbose=0, global_lr=1.0,
            global_momentum=0.9, min_bucket_size=500, refit=False, **kwargs):
        """
            Fit a generic preference learning FATE-network model on a provided set of queries.

            The provided queries can be of a fixed size (numpy arrays) or of
            varying sizes in which case dictionaries are expected as input.

            For varying sizes a meta gradient descent is performed across the
            different query sizes.

            Parameters
            ----------
            X : numpy array or dict
                Feature vectors of the objects
                (n_instances, n_objects, n_features) if numpy array or map from n_objects to numpy arrays
            Y : numpy array or dict
                Preferences in form of rankings or choices for given objects
                (n_instances, n_objects) if numpy array or map from n_objects to numpy arrays
            epochs : int
                Number of epochs to run if training for a fixed query size or
                number of epochs of the meta gradient descent for the variadic model
            inner_epochs : int
                Number of epochs to train for each query size inside the variadic
                model
            callbacks : list
                List of callbacks to be called during optimization
            validation_split : float (range : [0,1])
                Percentage of instances to split off to validate on
            verbose : bool
                Print verbose information
            global_lr : float
                Learning rate of the meta gradient descent (variadic model only)
            global_momentum : float
                Momentum for the meta gradient descent (variadic model only)
            min_bucket_size : int
                Restrict the training to queries of a minimum size
            refit : bool
                If True, create a new model object, otherwise continue fitting the
                existing one if one exists.
            **kwargs :
                Keyword arguments for the fit function
        """
        self._fit(X=X, Y=Y, epochs=epochs, inner_epochs=inner_epochs, callbacks=callbacks,
                  validation_split=validation_split, verbose=verbose, global_lr=global_lr,
                  global_momentum=global_momentum, min_bucket_size=min_bucket_size, refit=refit, **kwargs)

    def fit_generator(self, generator, epochs=35, steps_per_epoch=10, inner_epochs=1, callbacks=None, verbose=0,
                      global_lr=1.0, global_momentum=0.9, min_bucket_size=500, refit=False, **kwargs):
        """
            Fit a generic object ranking FATE-network on a set of queries provided by
            a generator.

            The provided queries can be of a fixed size (numpy arrays) or of
            varying sizes in which case dictionaries are expected as input.

            For varying sizes a meta gradient descent is performed across the
            different query sizes.

            Parameters
            ----------
            generator :
                A generator or an instance of `Sequence` (:class:`keras.utils.Sequence`) object in order to avoid
                duplicate data when using multiprocessing.
                The output of the generator must be either
                    - a tuple `(inputs, targets)`
                    - a tuple `(inputs, targets, sample_weights)`.
                This tuple (a single output of the generator) makes a single batch.
                Therefore, all arrays in this tuple must have the same length (equal to the size of this batch).
                Different batches may have different sizes.
                For example, the last batch of the epoch is commonly smaller than the others, if the size of the dataset
                is not divisible by the batch size. The generator is expected to loop over its data indefinitely. An
                epoch finishes when `steps_per_epoch` batches have been seen by the model.
            epochs : int
                Number of epochs to run if training for a fixed query size or
                number of epochs of the meta gradient descent for the variadic model
            steps_per_epoch : int
                Number of batches to train per epoch
            inner_epochs : int
                Number of epochs to train for each query size inside the variadic
                model
            callbacks : list
                List of callbacks to be called during optimization
            verbose : bool
                Print verbose information
            global_lr : float
                Learning rate of the meta gradient descent (variadic model only)
            global_momentum : float
                Momentum for the meta gradient descent (variadic model only)
            min_bucket_size : int
                Restrict the training to queries of a minimum size
            refit : bool
                If True, create a new model object, otherwise continue fitting the
                existing one if one exists.
            **kwargs:
                Keyword arguments for the fit function
        """
        self._fit(generator=generator, epochs=epochs, steps_per_epoch=steps_per_epoch, inner_epochs=inner_epochs,
                  callbacks=callbacks, verbose=verbose, global_lr=global_lr, global_momentum=global_momentum,
                  min_bucket_size=min_bucket_size, refit=refit, **kwargs)

    def _get_context_representation(self, X, kwargs):
        n_objects = X.shape[-2]
        self.logger.info("Test Set instances {} objects {} features {}".format(*X.shape))
        input_layer_scorer = Input(shape=(n_objects, self.n_object_features), name="input_node")
        if self.n_hidden_set_layers >= 1:
            self.set_layer(input_layer_scorer)
            fr = self.set_layer.cached_models[n_objects].predict(X, **kwargs)
            del self.set_layer.cached_models[n_objects]
            X_n = np.empty((fr.shape[0], n_objects,
                            fr.shape[1] + self.n_object_features),
                           dtype="float")
            for i in range(n_objects):
                X_n[:, i] = np.concatenate((X[:, i], fr), axis=1)
            X = np.copy(X_n)
        return X

    def _predict_scores_fixed(self, X, **kwargs):
        """
            Predict the scores for a fixed ranking size.

            Parameters
            ----------
            X : numpy array
                float (n_instances, n_objects, n_features)

            Returns
            -------
            scores : numpy array
                float (n_instances, n_objects)

        """
        # model = self._construct_scoring_model(n_objects)
        X = self._get_context_representation(X, kwargs)
        n_instances, n_objects, n_features = X.shape
        self.logger.info("After applying the set representations features {}".format(n_features))
        input_layer_joint = Input(shape=(n_objects, n_features), name="input_joint_model")
        scores = []

        inputs = [create_input_lambda(i)(input_layer_joint) for i in
                  range(n_objects)]

        for i in range(n_objects):
            joint = inputs[i]
            for j in range(self.n_hidden_joint_layers):
                joint = self.joint_layers[j](joint)
            scores.append(self.scorer(joint))
        scores = concatenate(scores, name="final_scores")
        joint_model = Model(inputs=input_layer_joint, outputs=scores)
        predicted_scores = joint_model.predict(X)
        self.logger.info("Done predicting scores")
        return predicted_scores

    def clear_memory(self, n_objects=5, **kwargs):
        """
            Clear the memory, restores the currently fitted model back to prevent memory leaks.

            Parameters
            ----------
            n_objects : int
                float (n_instances, n_objects, n_features)
            **kwargs :
                Keyword arguments for the function
        """
        if self.hash_file is not None:
            self.model.save_weights(self.hash_file)
            K.clear_session()
            sess = tf.Session()
            K.set_session(sess)
            self.optimizer = self.optimizer.from_config(self._optimizer_config)
            self._construct_layers(activation=self.activation, kernel_initializer=self.kernel_initializer,
                                   kernel_regularizer=self.kernel_regularizer, **self.kwargs)
            self._create_set_layers(activation=self.activation, kernel_initializer=self.kernel_initializer,
                                    kernel_regularizer=self.kernel_regularizer, **self.kwargs)
            self.model = self.construct_model(self.n_object_features, n_objects)
            self.model.load_weights(self.hash_file)
        else:
            self.logger.info("Cannot clear the memory")

    def set_tunable_parameters(self, n_hidden_set_units=32, n_hidden_set_layers=2, n_hidden_joint_units=32,
                               n_hidden_joint_layers=2, reg_strength=1e-4, learning_rate=1e-3, batch_size=128, **point):
        """
            Set tunable parameters of the FATE-network to the values provided.

            Parameters
            ----------
            n_hidden_set_layers : int
                Number of hidden set layers
            n_hidden_set_units : int
                Number of hidden set units in each set layer
            n_hidden_joint_units: int
                Number of hidden joint layers
            n_hidden_joint_layers: int
                Number of hidden units in each joint layer
            reg_strength: float
                Regularization strength of the regularizer function applied to the `kernel` weights matrix
            learning_rate: float
                Learning rate of the stochastic gradient descent algorithm used by the network
            batch_size: int
                Batch size to use during training
            point: dict
                Dictionary containing parameter values which are not tuned for the network
        """
        FATENetworkCore.set_tunable_parameters(self, n_hidden_joint_units=n_hidden_joint_units,
                                               n_hidden_joint_layers=n_hidden_joint_layers, reg_strength=reg_strength,
                                               learning_rate=learning_rate, batch_size=batch_size, **point)

        self.n_hidden_set_units = n_hidden_set_units
        self.n_hidden_set_layers = n_hidden_set_layers

        self._create_set_layers(activation=self.activation, kernel_initializer=self.kernel_initializer,
                                kernel_regularizer=self.kernel_regularizer, **self.kwargs)
        if hasattr(self, 'model'):
            self.model = None
        if hasattr(self, 'models'):
            self.models = None
