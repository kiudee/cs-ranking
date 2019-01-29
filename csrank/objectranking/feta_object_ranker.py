import logging

from keras.optimizers import SGD
from keras.regularizers import l2

from csrank.feta_network import FETANetwork
from csrank.losses import hinged_rank_loss
from .object_ranker import ObjectRanker

__all__ = ['FETAObjectRanker']


class FETAObjectRanker(FETANetwork, ObjectRanker):
    def __init__(self, n_objects, n_object_features, n_hidden=2, n_units=8, add_zeroth_order_model=False,
                 max_number_of_objects=5, num_subsample=5, loss_function=hinged_rank_loss, batch_normalization=False,
                 kernel_regularizer=l2(l=1e-4), kernel_initializer='lecun_normal', activation='selu',
                 optimizer=SGD(lr=1e-4, nesterov=True, momentum=0.9), metrics=None, batch_size=256, random_state=None,
                 **kwargs):
        super().__init__(n_objects=n_objects, n_object_features=n_object_features, n_hidden=n_hidden, n_units=n_units,
                         add_zeroth_order_model=add_zeroth_order_model, max_number_of_objects=max_number_of_objects,
                         num_subsample=num_subsample, loss_function=loss_function,
                         batch_normalization=batch_normalization, kernel_regularizer=kernel_regularizer,
                         kernel_initializer=kernel_initializer, activation=activation, optimizer=optimizer,
                         metrics=metrics, batch_size=batch_size, random_state=random_state, **kwargs)
        self.logger = logging.getLogger(FETAObjectRanker.__name__)

    def fit(self, X, Y, **kwd):
        super().fit(X, Y, **kwd)

    def _predict_scores_fixed(self, X, **kwargs):
        return super()._predict_scores_fixed(X, **kwargs)

    def predict_scores(self, X, **kwargs):
        return super().predict_scores(X, **kwargs)

    def predict_for_scores(self, scores, **kwargs):
        return ObjectRanker.predict_for_scores(self, scores, **kwargs)

    def predict(self, X, **kwargs):
        return super().predict(X, **kwargs)

    def clear_memory(self, **kwargs):
        self.logger.info("Clearing memory")
        super().clear_memory(**kwargs)
