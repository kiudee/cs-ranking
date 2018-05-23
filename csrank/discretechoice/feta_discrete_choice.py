import logging

from keras.losses import binary_crossentropy
from keras.regularizers import l2

from csrank.objectranking.feta_ranker import FETAObjectRanker
from .discrete_choice import DiscreteObjectChooser


class FETADiscreteChoiceFunction(FETAObjectRanker, DiscreteObjectChooser):
    def __init__(self, n_objects, n_object_features, n_hidden=2, n_units=8,
                 add_zeroth_order_model=False, max_number_of_objects=5,
                 num_subsample=5, loss_function=binary_crossentropy,
                 batch_normalization=False, kernel_regularizer=l2(l=1e-4),
                 non_linearities='selu', optimizer="adam", metrics=None, batch_size=256,
                 random_state=None, **kwargs):
        super().__init__(n_objects, n_object_features, n_hidden, n_units, add_zeroth_order_model, max_number_of_objects,
                         num_subsample, loss_function, batch_normalization, kernel_regularizer, non_linearities,
                         optimizer, metrics, batch_size, random_state, **kwargs)
        self.logger = logging.getLogger(FETADiscreteChoiceFunction.__name__)

    def fit(self, X, Y, epochs=10, callbacks=None, validation_split=0.1, verbose=0, **kwd):
        super().fit(X, Y, epochs, callbacks, validation_split, verbose, **kwd)

    def predict_scores(self, X, **kwargs):
        return super().predict_scores(X, **kwargs)

    def predict(self, X, **kwargs):
        DiscreteObjectChooser.predict(X, **kwargs)
