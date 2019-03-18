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
           Create a FATENetwork architecture.
           Training and prediction complexity is linear in the number of objects.

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
            loss_function : function
                Differentiable loss function for the score vector
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
            metrics : list
                List of evaluation metrics (can be non-differentiable)
            random_state : int or object
                Numpy random state
            **kwargs
                Keyword arguments for the hidden set units
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
        """ Construct joint layers and [0,1] output nodes

        Connecting the layers is done in join_input_layers.

        Parameters
        ----------
        **kwargs
            Keyword arguments passed into the joint layers
        """
        self.logger.info(
            "Construct joint layers hidden units {} and layers {} ".format(
                self.n_hidden_joint_units,
                self.n_hidden_joint_layers))
        # Create joint hidden layers:
        self.joint_layers = []
        for i in range(self.n_hidden_joint_layers):
            self.joint_layers.append(Dense(self.n_hidden_joint_units, name="joint_layer_{}".format(i), **kwargs))
        self.logger.info('Construct output score node')
        self.scorer = Dense(1, name="output_node", activation='sigmoid', kernel_regularizer=self.kernel_regularizer)

    def fit(self, X, Y, tune_size=0.1, thin_thresholds=1, **kwargs):
        if tune_size > 0:
            X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=tune_size)
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
