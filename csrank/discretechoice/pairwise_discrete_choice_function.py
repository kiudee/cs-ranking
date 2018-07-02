import logging

from csrank.dataset_reader.discretechoice.util import generate_complete_pairwise_dataset
from csrank.discretechoice.discrete_choice import DiscreteObjectChooser
from csrank.objectranking.rank_svm import RankSVM


class RankSVMDCM(RankSVM, DiscreteObjectChooser):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.logger = logging.getLogger(RankSVMDCM.__name__)

    def convert_instances(self, X, Y):
        self.logger.debug('Creating the Dataset')
        garbage, garbage, x_train, garbage, y_single = generate_complete_pairwise_dataset(X, Y)
        del garbage
        assert x_train.shape[1] == self.n_object_features
        self.logger.debug('Finished the Dataset with instances {}'.format(x_train.shape[0]))
        return x_train, y_single

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
        self.logger.info("Clearing memory")
        super().clear_memory(**kwargs)
