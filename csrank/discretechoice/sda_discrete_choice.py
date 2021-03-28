from keras import Model
from keras.layers import Activation, Input
from keras.optimizers import SGD

from csrank.core.sda_core import SDACore
from csrank.discretechoice.discrete_choice import DiscreteObjectChooser


class SDADiscreteChoiceFunction(SDACore, DiscreteObjectChooser):
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
        dropout_rate=None,
        batch_size=128,
        activation="tanh",
        loss_function="categorical_hinge",
        metrics=["categorical_accuracy"],
        optimizer=SGD(lr=1e-4, nesterov=True, momentum=0.9, clipnorm=1.0),
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
            dropout_rate=dropout_rate,
            batch_size=batch_size,
            activation=activation,
            loss_function=loss_function,
            metrics=metrics,
            optimizer=optimizer,
            **kwargs
        )

    def construct_model(self, n_features, n_objects):
        model = super().construct_model(n_features=n_features, n_objects=n_objects)
        input_layer = Input(shape=(n_objects, n_features), name="input_node_outer")
        scores = Activation("sigmoid")(model(input_layer))
        final_model = Model(inputs=input_layer, outputs=scores)
        final_model.compile(
            loss=self.loss_function, optimizer=self.optimizer, metrics=self.metrics
        )
        return final_model

    def predict_scores(self, X, **kwargs):
        return super().predict_scores(X, **kwargs)

    def predict(self, X, **kwargs):
        return super().predict(X, **kwargs)

    def predict_for_scores(self, scores, **kwargs):
        return DiscreteObjectChooser.predict_for_scores(self, scores)
