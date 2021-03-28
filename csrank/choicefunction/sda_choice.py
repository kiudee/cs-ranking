from keras import Model
from keras.layers import Activation, Input
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split

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
        dropout_rate=None,
        batch_size=128,
        activation="tanh",
        loss_function="binary_crossentropy",
        metrics=None,
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
        self.threshold = threshold

    def fit(
        self,
        X,
        Y,
        epochs=10,
        callbacks=None,
        validation_split=0.1,
        tune_size=0.1,
        thin_thresholds=1,
        verbose=0,
        **kwd
    ):
        if tune_size > 0:
            X_train, X_val, Y_train, Y_val = train_test_split(
                X, Y, test_size=tune_size, random_state=self.random_state
            )
            try:
                super().fit(
                    X_train,
                    Y_train,
                    epochs=epochs,
                    callbacks=callbacks,
                    validation_split=validation_split,
                    verbose=verbose,
                    **kwd
                )
            finally:
                self.logger.info(
                    "Fitting utility function finished. Start tuning threshold."
                )
                self.threshold = self._tune_threshold(
                    X_val, Y_val, thin_thresholds=thin_thresholds, verbose=verbose
                )
        else:
            super().fit(
                X,
                Y,
                epochs=epochs,
                callbacks=callbacks,
                validation_split=validation_split,
                verbose=verbose,
                **kwd
            )
            self.threshold = 0.5

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
        return ChoiceFunctions.predict_for_scores(self, scores)
