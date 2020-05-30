import logging

from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.layers import Dense
from keras.layers import Input
from keras.layers import Lambda
from keras.layers.merge import average
from keras.models import Model

__all__ = ["NormalizedDense", "DeepSet", "create_input_lambda"]


class NormalizedDense(object):
    """Stop training when a monitored quantity has stopped improving.
        # Arguments
        units: Positive integer, dimensionality of the output space.
        activation: Activation function to use
            (see [activations](../activations.md)).
            If you don't specify anything, no activation is applied
            (ie. "relu:).
        normalize_before_activation: True if normalize the inputs before applying the activation.
        False if activation is applied before Bach Normalization
        """

    def __init__(
        self, units, activation="relu", normalize_before_activation=False, **kwd
    ):
        self.dense = Dense(units, activation="linear", **kwd)
        self.activation = Activation(activation=activation)
        self.batchnorm = BatchNormalization()
        self.norm_layer = None
        self.normalize_before_activation = normalize_before_activation

    def __call__(self, x):
        if self.normalize_before_activation:
            return self.activation(self.batchnorm(self.dense(x)))
        else:
            return self.batchnorm(self.activation(self.dense(x)))

    def get_weights(self):
        w_b = self.batchnorm.get_weights()
        w_d = self.dense.get_weights()
        return w_b, w_d

    def set_weights(self, weights):
        w_b, w_d = weights
        self.batchnorm.set_weights(w_b)
        self.dense.set_weights(w_d)


class DeepSet(object):
    """Deep layer for learning representations for sets of objects.

    Parameters
    ----------
    units : int
        Number of units in each representation layer

    layers : int
        Number of layers to use for learning the representation

    activation : string, optional (default='selu')
        Activation function to use in each unit

    kernel_initializer : string, optional (default='lecun_normal')
        Initializer for the weight matrix

    input_shape : array_like
        Should provide (n_objects, n_features) (DEPRECATED)

    Attributes
    ----------
    model : Keras model
        Representing the complete deep set layer

    set_mapping_layers : list
        List of densely connected hidden layers
    """

    def __init__(
        self,
        units,
        layers=1,
        activation="selu",
        kernel_initializer="lecun_normal",
        kernel_regularizer=None,
        input_shape=None,
        **kwargs,
    ):
        self.logger = logging.getLogger("DeepSets")
        self.n_units = units
        if input_shape is not None:
            self.logger.warning(
                "input_shape is deprecated, since the number "
                "of objects is now inferred"
            )
            self.n_features = input_shape[1]
        self.n_layers = layers
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer

        self.cached_models = dict()
        self._construct_layers(
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            activation=activation,
            **kwargs,
        )

    def _construct_layers(self, **kwargs):
        # Create set representation layers:
        self.set_mapping_layers = []
        for i in range(self.n_layers):
            self.set_mapping_layers.append(
                Dense(self.n_units, name="set_layer_{}".format(i), **kwargs)
            )

    def _create_model(self, shape):
        n_objects, n_features = shape[1].value, shape[2].value
        if hasattr(self, "n_features"):
            if self.n_features != n_features:
                self.logger.error("Number of features is not consistent.")
        input_layer = Input(shape=(n_objects, n_features))
        inputs = [create_input_lambda(i)(input_layer) for i in range(n_objects)]

        # Connect input tensors with set mapping layer:
        set_mappings = []
        for i in range(n_objects):
            curr = inputs[i]
            for j in range(len(self.set_mapping_layers)):
                curr = self.set_mapping_layers[j](curr)
            set_mappings.append((i, curr))

        # TODO: is feature_repr used outside?
        feature_repr = average([x for (j, x) in set_mappings])

        self.cached_models[n_objects] = Model(inputs=input_layer, outputs=feature_repr)

    def __call__(self, x):
        shape = x.shape
        n_objects = shape[1].value
        if n_objects not in self.cached_models:
            self._create_model(shape)
        return self.cached_models[n_objects](x)

    def get_weights(self):
        w_set = [x.get_weights() for x in self.set_mapping_layers]
        return w_set

    def set_weights(self, weights):
        for i, layer in enumerate(self.set_mapping_layers):
            layer.set_weights(weights[i])


def create_input_lambda(i):
    """Extracts off an object tensor from an input tensor"""
    return Lambda(lambda x: x[:, i])
