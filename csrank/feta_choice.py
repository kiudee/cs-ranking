from csrank import FETANetwork

from keras.losses import binary_crossentropy
from keras.regularizers import l2


class FETAChoiceFunction(FETANetwork):
    def __init__(self, n_objects, n_features, n_hidden=2, n_units=8,
                 add_zeroth_order_model=False, max_number_of_objects=5,
                 num_subsample=5, loss_function=binary_crossentropy,
                 batch_normalization=False, kernel_regularizer=l2(l=1e-4),
                 non_linearities='selu', optimizer="adam", metrics=None,
                 use_early_stopping=False, es_patience=300, batch_size=256,
                 random_state=None, **kwargs):
        super().__init__(n_objects, n_features, n_hidden, n_units,
                         add_zeroth_order_model, max_number_of_objects,
                         num_subsample, loss_function, batch_normalization,
                         kernel_regularizer, non_linearities, optimizer,
                         metrics, use_early_stopping, es_patience, batch_size,
                         random_state, **kwargs)

    def fit(self, X, Y, epochs=10, callbacks=None, validation_split=0.1, verbose=0, **kwd):
        super().fit(X, Y, epochs, callbacks, validation_split, verbose, **kwd)

    def predict_scores(self, X, **kwargs):
        return super().predict_scores(X, **kwargs)

    def predict(self, X, **kwargs):
        return super().predict(X, **kwargs)
