import logging

from keras.layers import Dense
from keras.optimizers import SGD
from keras.regularizers import l2

from csrank.discretechoice.discrete_choice import DiscreteObjectChooser
from csrank.core.fate_network import FATENetwork


class FATEDiscreteChoiceFunction(FATENetwork, DiscreteObjectChooser):
    def __init__(self, n_object_features, n_hidden_set_layers=2, n_hidden_set_units=2, loss_function='categorical_hinge'
                 , metrics=['categorical_accuracy'], n_hidden_joint_layers=32,
                 n_hidden_joint_units=32, activation='selu', kernel_initializer='lecun_normal',
                 kernel_regularizer=l2(l=0.01), optimizer=SGD(lr=1e-4, nesterov=True, momentum=0.9), batch_size=256,
                 random_state=None, **kwargs):
        self.loss_function = loss_function
        self.metrics = metrics
        super().__init__(n_object_features=n_object_features, n_hidden_set_layers=n_hidden_set_layers,
                         n_hidden_set_units=n_hidden_set_units, n_hidden_joint_layers=n_hidden_joint_layers,
                         n_hidden_joint_units=n_hidden_joint_units, activation=activation,
                         kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer,
                         optimizer=optimizer, batch_size=batch_size, random_state=random_state, **kwargs)
        self.logger = logging.getLogger(FATEDiscreteChoiceFunction.__name__)

    def _construct_layers(self, **kwargs):
        """ Construct basic layers shared by all ranking algorithms:
         * Joint dense hidden layers
         * Output scoring layer

        Connecting the layers is done in join_input_layers and will be done in
        implementing classes.
        """
        self.logger.info("Construct joint layers hidden units {} and layers {} ".format(self.n_hidden_joint_units,
                                                                                        self.n_hidden_joint_layers))
        # Create joint hidden layers:
        self.joint_layers = []
        for i in range(self.n_hidden_joint_layers):
            self.joint_layers.append(Dense(self.n_hidden_joint_units, name="joint_layer_{}".format(i), **kwargs))

        self.logger.info('Construct output score node')
        self.scorer = Dense(1, name="output_node", activation='sigmoid', kernel_regularizer=self.kernel_regularizer)

    def fit(self, X, Y, **kwd):
        super().fit(X, Y, **kwd)

    def _predict_scores_fixed(self, X, **kwargs):
        return super()._predict_scores_fixed(X, **kwargs)

    def predict_scores(self, X, **kwargs):
        return super().predict_scores(X, **kwargs)

    def predict_for_scores(self, scores, **kwargs):
        return DiscreteObjectChooser.predict_for_scores(self, scores, **kwargs)

    def predict(self, X, **kwargs):
        return super().predict(X, **kwargs)

    def clear_memory(self, **kwargs):
        self.logger.info("Clearing memory")
        super().clear_memory(**kwargs)
