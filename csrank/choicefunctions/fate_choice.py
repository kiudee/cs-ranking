import logging

from keras.layers import Dense
from keras.losses import binary_crossentropy
from keras.optimizers import SGD
from keras.regularizers import l2
from sklearn.model_selection import train_test_split

from csrank.core.fate_network import FATENetwork
from .choice_functions import ChoiceFunctions


class FATEChoiceFunction(FATENetwork, ChoiceFunctions):
    def __init__(self, n_object_features, n_hidden_set_layers=2, n_hidden_set_units=2, n_hidden_joint_layers=32,
                 n_hidden_joint_units=32, loss_function=binary_crossentropy, activation='selu',
                 kernel_initializer='lecun_normal', kernel_regularizer=l2(l=0.01),
                 optimizer=SGD(lr=1e-4, nesterov=True, momentum=0.9), batch_size=256, metrics=None, random_state=None,
                 **kwargs):
        """
            Create a FATE-network architecture for leaning discrete choice function. The first-aggregate-then-evaluate
            approach learns an embedding of each object and then aggregates that into a context representation
            :math:`\\mu_{C(x)}` and then scores each object :math:`x` using a generalized utility function
            :math:`U (x, \\mu_{C(x)})`.
            To make it computationally efficient we take the the context :math:`C(x)` as query set :math:`Q`.
            The context-representation is evaluated as:

            .. math::
                \\mu_{C(x)} = \\frac{1}{\\lvert C(x) \\lvert} \\sum_{y \\in C(x)} \\phi(y)

            where :math:`\phi \colon \mathcal{X} \\to \mathcal{Z}` maps each object :math:`y` to an
            :math:`m`-dimensional embedding space :math:`\mathcal{Z} \subseteq \mathbb{R}^m`.
            The choice set is defined as:

            .. math::

                c(Q) = \{ x \in Q \lvert \, U (x, \\mu_{C(x)}) > t \}


            Parameters
            ----------
            n_object_features : int
                Dimensionality of the feature space of each object
            n_hidden_set_layers : int
                Number of set layers.
            n_hidden_set_units : int
                Number of hidden set units.
            n_hidden_joint_layers : int
                Number of joint layers.
            n_hidden_joint_units : int
                Number of joint units.
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
            loss_function : function
                Differentiable loss function for the score vector
            metrics : list
                List of evaluation metrics (can be non-differentiable)
            random_state : int or object
                Numpy random state
            **kwargs
                Keyword arguments for the @FATENetwork
        """
        self.loss_function = loss_function
        self.metrics = metrics
        super().__init__(n_object_features=n_object_features, n_hidden_set_layers=n_hidden_set_layers,
                         n_hidden_set_units=n_hidden_set_units, n_hidden_joint_layers=n_hidden_joint_layers,
                         n_hidden_joint_units=n_hidden_joint_units, activation=activation,
                         kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer,
                         optimizer=optimizer, batch_size=batch_size, random_state=random_state, **kwargs)
        self.logger = logging.getLogger(FATEChoiceFunction.__name__)
        self.threshold = 0.5

    def _construct_layers(self, **kwargs):
        """
            Construct basic layers shared by all the objects:
                * Joint dense hidden layers
                * Output scoring layer is sigmoid output for choice model

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
        self.scorer = Dense(1, name="output_node", activation='sigmoid', kernel_regularizer=self.kernel_regularizer)

    def construct_model(self, n_features, n_objects):
        return super().construct_model(n_features, n_objects)

    def fit(self, X, Y, epochs=35, inner_epochs=1, callbacks=None, validation_split=0.1, verbose=0, global_lr=1.0,
            global_momentum=0.9, min_bucket_size=500, refit=False, tune_size=0.1, thin_thresholds=1, **kwargs):
        """
            Fit a generic FATE-network model for learning a choice function on a provided set of queries.

            The provided queries can be of a fixed size (numpy arrays) or of varying sizes in which case dictionaries
            are expected as input. For varying sizes a meta gradient descent is performed across the
            different query sizes.

            Parameters
            ----------
            X : numpy array or dict
                Feature vectors of the objects
                (n_instances, n_objects, n_features) if numpy array or map from n_objects to numpy arrays
            Y : numpy array or dict
                Choices for given objects in the query
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
            tune_size: float (range : [0,1])
                Percentage of instances to split off to tune the threshold for the choice function
            thin_thresholds: int
                The number of instances of scores to skip while tuning the threshold
            **kwargs :
                Keyword arguments for the fit function
        """
        if tune_size > 0:
            X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=tune_size, random_state=self.random_state)
            try:
                super().fit(X_train, Y_train, **kwargs)
            finally:
                self.logger.info('Fitting utility function finished. Start tuning threshold.')
                self.threshold = self._tune_threshold(X_val, Y_val, thin_thresholds=thin_thresholds)
        else:
            super().fit(X, Y, **kwargs)
            self.threshold = 0.5

    def _predict_scores_fixed(self, X, **kwargs):
        return super()._predict_scores_fixed(X, **kwargs)

    def predict_scores(self, X, **kwargs):
        return super().predict_scores(X, **kwargs)

    def predict_for_scores(self, scores, **kwargs):
        return ChoiceFunctions.predict_for_scores(self, scores, **kwargs)

    def predict(self, X, **kwargs):
        return super().predict(X, **kwargs)

    def clear_memory(self, **kwargs):
        self.logger.info("Clearing memory")
        super().clear_memory(**kwargs)

    def set_tunable_parameters(self, n_hidden_set_units=32, n_hidden_set_layers=2, n_hidden_joint_units=32,
                               n_hidden_joint_layers=2, reg_strength=1e-4, learning_rate=1e-3, batch_size=128, **point):
        super().set_tunable_parameters(n_hidden_set_units=n_hidden_set_units, n_hidden_set_layers=n_hidden_set_layers,
                                       n_hidden_joint_units=n_hidden_joint_units,
                                       n_hidden_joint_layers=n_hidden_joint_layers, reg_strength=reg_strength,
                                       learning_rate=learning_rate, batch_size=batch_size, **point)
