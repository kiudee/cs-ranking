import logging

from csrank.cmpnet_core import CmpNetCore
from csrank.dataset_reader.discretechoice.util import generate_complete_pairwise_dataset
from csrank.discretechoice.discrete_choice import DiscreteObjectChooser


class CmpNetDiscreteChoiceFunction(CmpNetCore, DiscreteObjectChooser):
    def __init__(self, loss_function='binary_crossentropy', metrics=['binary_accuracy'],
                 **kwargs):
        super().__init__(loss_function=loss_function, metrics=metrics, **kwargs)
        self.logger = logging.getLogger(CmpNetDiscreteChoiceFunction.__name__)
        self.logger.info("Initializing network with object features {}".format(self.n_object_features))

    def convert_instances(self, X, Y):
        self.logger.debug('Creating the Dataset')
        x1, x2, y_double, garbage = generate_complete_pairwise_dataset(X, Y)
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
        return DiscreteObjectChooser.predict_for_scores(self, scores, **kwargs)

    def predict(self, X, **kwargs):
        return super().predict(X, **kwargs)

    def clear_memory(self, **kwargs):
        super().clear_memory(**kwargs)
