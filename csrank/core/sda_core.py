import logging

import tensorflow as tf
from keras import Input, Model
from keras import backend as K
from keras import optimizers
from keras.regularizers import l2

from csrank.layers import DeepSetSDA
from csrank.learner import Learner


def kinked_tanh(x, slope=1.5):
    return tf.tanh(x) * (
        tf.cast(x < 0, dtype=tf.float32) * slope + tf.cast(x >= 0, dtype=tf.float32)
    )


class SDACore(Learner):
    def __init__(
        self,
        n_features,
        tanh_slope=1.5,
        n_linear_units=24,
        n_w_units=16,
        n_w_layers=2,
        n_r_units=16,
        n_r_layers=2,
        learning_rate=1e-3,
        regularization_strength=1e-4,
        batch_size=128,
        activation="tanh",
        loss_function="mse",
        metrics=None,
        optimizer="adam",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.logger = logging.getLogger(SDACore.__name__)
        self.n_features = n_features
        self.tanh_slope = tanh_slope
        self.n_linear_units = n_linear_units
        self.n_w_units = n_w_units
        self.n_w_layers = n_w_layers
        self.n_r_units = n_r_units
        self.n_r_layers = n_r_layers
        self.batch_size = batch_size
        self.activation = activation
        self.loss_function = loss_function
        self.metrics = metrics
        self.optimizer = optimizers.get(optimizer)
        self.learning_rate = learning_rate
        K.set_value(self.optimizer.lr, self.learning_rate)
        self._optimizer_config = self.optimizer.get_config()
        self.regularization_strength = l2(regularization_strength)
        self.model = None
        self._construct_layers(**kwargs)

    def _construct_layers(self, **kwargs):
        # ell x linear layer f_i
        self.linear_embeddings = tf.keras.layers.Dense(
            units=self.n_linear_units,
            activation="linear",
            input_shape=(self.n_features,),
            kernel_regularizer=self.regularization_strength,
        )
        # 2x set NN (2 hidden layers with 16 units)
        self.w_network = DeepSetSDA(
            output_dim=1,
            input_shape=(1,),
            set_layers=self.n_w_layers,
            set_units=self.n_w_units,
            activation=self.activation,
            kernel_regularizer=self.regularization_strength,
        )
        self.r_network = DeepSetSDA(
            output_dim=1,
            input_shape=(1,),
            set_layers=self.n_r_layers,
            set_units=self.n_r_units,
            activation=self.activation,
            kernel_regularizer=self.regularization_strength,
        )

    def construct_model(self, n_features, n_objects):
        input_layer = Input(shape=(n_objects, n_features), name="input_node")
        lin_scores = self.linear_embeddings(input_layer)

        # Reshape to (n_batch, n_linear_units, n_objects, 1)
        lin_scores = tf.transpose(tf.expand_dims(lin_scores, -1), perm=[0, 2, 1, 3])

        # Compute set scores (n_batch, n_linear_units, 1, 1)
        w = tf.expand_dims(self.w_network(lin_scores), -1)
        r = tf.expand_dims(self.r_network(lin_scores), -1)

        # (n_batch, n_linear_units, n_objects, 1)
        mu = kinked_tanh(lin_scores - r, slope=self.tanh_slope)

        # result (n_batch, n_objects)
        scores = tf.reduce_sum(w * mu, axis=(-1, -3))

        model = Model(inputs=input_layer, outputs=scores)
        model.compile(
            loss=self.loss_function, optimizer=self.optimizer, metrics=self.metrics
        )
        return model

    def clear_memory(self, n_objects, **kwargs):
        weights = self.model.get_weights()
        K.clear_session()
        sess = tf.Session()
        K.set_session(sess)
        self.optimizer = self.optimizer.from_config(self._optimizer_config)
        self._construct_layers()
        self.model = self.construct_model(self.n_features, n_objects)
        self.model.set_weights(weights)

    def fit(self, X, Y, batch_size=None, **kwargs):
        n_instances, n_objects, n_features = X.shape
        self.model = self.construct_model(n_objects=n_objects, n_features=n_features)
        self.logger.info("Fitting the model")
        if batch_size is None:
            batch_size = self.batch_size
        self.model.fit(x=X, y=Y, batch_size=batch_size, **kwargs)
        self.logger.info("Fitting complete")

    def _predict_scores_fixed(self, X, **kwargs):
        return self.model.predict(X)

    def predict_for_scores(self, scores, **kwargs):
        return scores

    def predict_scores(self, X, **kwargs):
        return super().predict_scores(X, **kwargs)

    def predict(self, X, **kwargs):
        return super().predict(X, **kwargs)

    def get_weights(self, **kwargs):
        if self.model is not None:
            return self.model.get_weights()
        return None

    def set_weights(self, weights, **kwargs):
        if self.model is not None:
            self.model.set_weights(weights)
        raise AttributeError("No model has been fit yet.")

    def set_tunable_parameters(
        self,
        learning_rate=1e-3,
        batch_size=128,
        regularization_strength=1e-4,
        tanh_slope=1.5,
        n_linear_units=24,
        n_w_units=16,
        n_w_layers=2,
        n_r_units=16,
        n_r_layers=2,
        **point
    ):
        self.tanh_slope = tanh_slope
        self.n_linear_units = n_linear_units
        self.n_w_units = n_w_units
        self.n_w_layers = n_w_layers
        self.n_r_units = n_r_units
        self.n_r_layers = n_r_layers
        self.batch_size = batch_size
        self.regularization_strength = l2(regularization_strength)

        self.optimizer = self.optimizer.from_config(self._optimizer_config)
        K.set_value(self.optimizer.lr, learning_rate)

        self._construct_layers()
        if hasattr(self, "model"):
            self.model = None
