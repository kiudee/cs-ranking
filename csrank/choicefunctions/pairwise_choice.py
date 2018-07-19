import logging

from sklearn.model_selection import train_test_split

from csrank.choicefunctions.util import generate_complete_pairwise_dataset
from csrank.objectranking.rank_svm import RankSVM
from .choice_functions import ChoiceFunctions


class RankSVMChoiceFunction(RankSVM, ChoiceFunctions):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.logger = logging.getLogger(RankSVMChoiceFunction.__name__)
        self.threshold = 0.5

    def convert_instances(self, X, Y):
        self.logger.debug('Creating the Dataset')
        garbage, garbage, x_train, garbage, y_single = generate_complete_pairwise_dataset(X, Y)
        del garbage
        assert x_train.shape[1] == self.n_object_features
        self.logger.debug('Finished the Dataset with instances {}'.format(x_train.shape[0]))
        return x_train, y_single

    def fit(self, X, Y, tune_size=0.1, thin_thresholds=1, **kwd):
        if tune_size > 0:
            X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=tune_size)
            try:
                super().fit(X_train, Y_train, **kwd)
            finally:
                self.logger.info('Fitting utility function finished. Start tuning threshold.')
                self.threshold = self._tune_threshold(X_val, Y_val, thin_thresholds=thin_thresholds)
        else:
            super().fit(X, Y, **kwd)
            self.threshold = 0.5

    def _predict_scores_fixed(self, X, **kwargs):
        return super()._predict_scores_fixed(X, **kwargs)

    def predict_scores(self, X, **kwargs):
        return super().predict_scores(X, **kwargs)

    def predict_for_scores(self, scores, **kwargs):
        return ChoiceFunctions.predict_for_scores(self, scores, **kwargs)

    def predict(self, X, **kwargs):
        return super().predict(X, **kwargs)

    def clear_memory(self, **kwargs):
        self.logger.info("Clearing memory")
        super().clear_memory(**kwargs)
