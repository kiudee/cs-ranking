import logging

from keras.layers import Dense
from keras.losses import binary_crossentropy

from csrank.fate_ranking import FATEObjectRankingCore


class FATEChoiceFunction(FATEObjectRankingCore):
    def __init__(self,
                 n_object_features,
                 n_hidden_joint_layers=2,
                 n_hidden_joint_units=32,
                 n_hidden_set_layers=2,
                 n_hidden_set_units=32,
                 loss_function=binary_crossentropy,
                 metrics=None,
                 **kwargs):
        super().__init__(n_object_features=n_object_features,
                         n_hidden_joint_layers=n_hidden_joint_layers,
                         n_hidden_joint_units=n_hidden_joint_units,
                         n_hidden_set_layers=n_hidden_set_layers,
                         n_hidden_set_units=n_hidden_set_units,
                         metrics=metrics,
                         **kwargs)
        self.loss_function = loss_function
        self.metrics = metrics
        self.logger = logging.Logger('FATEChoiceFunction')

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
            self.joint_layers.append(
                Dense(self.n_hidden_joint_units,
                      name="joint_layer_{}".format(i),
                      **kwargs)
            )

        self.logger.info('Construct output score node')
        self.scorer = Dense(1, name="output_node", activation='sigmoid',
                            kernel_regularizer=self.kernel_regularizer)

    def fit(self, X, Y, **kwargs):
        super().fit(X, Y, **kwargs)

    def predict_scores(self, X, **kwargs):
        return super().predict_scores(X, **kwargs)

    def predict(self, X, **kwargs):
        scores = self.predict_scores(X, **kwargs)
        return scores > 0.5
