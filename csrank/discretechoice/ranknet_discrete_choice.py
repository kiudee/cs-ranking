import logging

from csrank.dataset_reader.discretechoice.util import generate_complete_pairwise_dataset
from csrank.objectranking.rank_net import RankNet
from csrank.rank_network import RankNetwork
from .discrete_choice import DiscreteObjectChooser


class RankNetDiscreteChoiceFunction(RankNetwork, DiscreteObjectChooser):
    def __init__(self, loss_function='categorical_hinge', metrics=None, **kwargs):
        RankNet.__init__(self, **kwargs)
        self.logger = logging.getLogger(RankNetDiscreteChoiceFunction.__name__)
        self.loss_function = loss_function
        if metrics is None:
            metrics = ['categorical_accuracy']
        self.metrics = metrics

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
        return super().predict_scores(self, X, **kwargs)

    def predict(self, X, **kwargs):
        return super().predict(self, X, **kwargs)

    def predict_for_scores(self, scores, **kwargs):
        return DiscreteObjectChooser.predict_for_scores(self, scores, **kwargs)

    def _predict_scores_fixed(self, X, **kwargs):
        return super()._predict_scores_fixed(self, X, **kwargs)

    def clear_memory(self, **kwargs):
        super().clear_memory(self, **kwargs)
