import logging

from keras.losses import binary_crossentropy
from keras.regularizers import l2

from csrank.objectranking.rank_net import RankNet
from csrank.dataset_reader.discretechoice.util import generate_complete_pairwise_dataset
from csrank.discretechoice.discrete_choice import DiscreteObjectChooser


class RankNetDiscreteChoiceFunction(RankNet, DiscreteObjectChooser):
    def __init__(self, n_objects, n_object_features, n_hidden=2, n_units=8, loss_function=binary_crossentropy,
                 batch_normalization=False, kernel_regularizer=l2(l=1e-4),
                 non_linearities='selu', optimizer="adam", metrics=None, batch_size=256,
                 random_state=None, **kwargs):
        super().__init__(n_objects, n_object_features, n_hidden, n_units, loss_function, batch_normalization,
                         kernel_regularizer, non_linearities, optimizer, metrics, batch_size, random_state, **kwargs)
        self.logger = logging.getLogger(RankNetDiscreteChoiceFunction.__name__)

    def fit(self, X, Y, epochs=10, callbacks=None, validation_split=0.1, verbose=0, **kwd):
        super().fit(X, Y, epochs=epochs, callbacks=callbacks, validation_split=validation_split, verbose=verbose)

    def convert_instances(self, X, Y):
        self.logger.debug('Creating the Dataset')
        X1, X2, garbage, Y_single = generate_complete_pairwise_dataset(X, Y)
        del garbage
        if X1.shape[0] > self.threshold_instances:
            indices = self.random_state.choice(X1.shape[0], self.threshold_instances, replace=False)
            X1 = X1[indices, :]
            X2 = X2[indices, :]
            Y_single = Y_single[indices]
        self.logger.debug('Finished the Dataset')
        return X1, X2, Y_single

    def predict_scores(self, X, **kwargs):
        return super().predict_scores(X, **kwargs)

    def predict(self, X, **kwargs):
        DiscreteObjectChooser.predict(X, **kwargs)
