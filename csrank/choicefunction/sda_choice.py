from keras import Model
from keras.layers import Dense, Input

from csrank.choicefunction.choice_functions import ChoiceFunctions
from csrank.core.sda_core import SDACore


class SDAChoiceFunction(SDACore, ChoiceFunctions):
    def __init__(
        self,
        n_features,
        threshold=0.5,
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
        loss_function="binary_crossentropy",
        metrics=None,
        optimizer="adam",
        **kwargs
    ):
        super().__init__(
            n_features=n_features,
            tanh_slope=tanh_slope,
            n_linear_units=n_linear_units,
            n_w_units=n_w_units,
            n_w_layers=n_w_layers,
            n_r_units=n_r_units,
            n_r_layers=n_r_layers,
            learning_rate=learning_rate,
            regularization_strength=regularization_strength,
            batch_size=batch_size,
            activation=activation,
            loss_function=loss_function,
            metrics=metrics,
            optimizer=optimizer,
            **kwargs
        )
        self.threshold = threshold

    def _construct_layers(self, **kwargs):
        super()._construct_layers(**kwargs)
        self.scorer = Dense(
            units=1,
            name="output_node",
            activation="sigmoid",
            kernel_regularizer=self.regularization_strength,
        )

    def construct_model(self, n_features, n_objects):
        model = super().construct_model(n_features=n_features, n_objects=n_objects)
        input_layer = Input(shape=(n_objects, n_features), name="input_node_outer")
        scores = model(input_layer)
        sigmoid_scores = self.scorer(scores)
        final_model = Model(inputs=input_layer, outputs=sigmoid_scores)
        final_model.compile(
            loss=self.loss_function, optimizer=self.optimizer, metrics=self.metrics
        )
        return final_model

    def predict_scores(self, X, **kwargs):
        return super().predict_scores(X, **kwargs)

    def predict(self, X, **kwargs):
        return super().predict(X, **kwargs)
