import logging

from keras.optimizers import SGD
from keras.regularizers import l2

from csrank.core.cmpnet_core import CmpNetCore
from csrank.dataset_reader.objectranking.util import generate_complete_pairwise_dataset
from csrank.objectranking.object_ranker import ObjectRanker

__all__ = ['CmpNet']


class CmpNet(CmpNetCore, ObjectRanker):
    def __init__(self, n_object_features, n_hidden=2, n_units=8, loss_function='binary_crossentropy',
                 batch_normalization=True, kernel_regularizer=l2(l=1e-4), kernel_initializer='lecun_normal',
                 activation='relu', optimizer=SGD(lr=1e-4, nesterov=True, momentum=0.9), metrics=['binary_accuracy'],
                 batch_size=256, random_state=None, **kwargs):
        super().__init__(n_object_features=n_object_features, n_hidden=n_hidden, n_units=n_units,
                         loss_function=loss_function, batch_normalization=batch_normalization,
                         kernel_regularizer=kernel_regularizer, kernel_initializer=kernel_initializer,
                         activation=activation, optimizer=optimizer, metrics=metrics, batch_size=batch_size,
                         random_state=random_state, **kwargs)
        self.logger = logging.getLogger(CmpNet.__name__)
        self.logger.info("Initializing network with object features {}".format(self.n_object_features))

    def convert_instances(self, X, Y):
        self.logger.debug('Creating the Dataset')
        garbage, x1, x2, y_double, garbage = generate_complete_pairwise_dataset(X, Y)
        del garbage
        self.logger.debug('Finished the Dataset')
        if x1.shape[0] > self.threshold_instances:
            indices = self.random_state.choice(x1.shape[0], self.threshold_instances, replace=False)
            x1 = x1[indices, :]
            x2 = x2[indices, :]
            y_double = y_double[indices, :]
        return x1, x2, y_double

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
        super().clear_memory(**kwargs)
