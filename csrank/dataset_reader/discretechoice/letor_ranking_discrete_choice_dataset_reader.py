import logging

from sklearn.utils import check_random_state

from csrank.constants import DISCRETE_CHOICE
from csrank.dataset_reader.util import standardize_features
from .util import sub_sampling_discrete_choices_from_relevance
from ..letor_ranking_dataset_reader import LetorRankingDatasetReader


class LetorRankingDiscreteChoiceDatasetReader(LetorRankingDatasetReader):
    def __init__(self, random_state=None, n_objects=5, **kwargs):
        super(LetorRankingDiscreteChoiceDatasetReader, self).__init__(learning_problem=DISCRETE_CHOICE, **kwargs)
        self.logger = logging.getLogger(LetorRankingDiscreteChoiceDatasetReader.__name__)
        self.random_state = check_random_state(random_state)
        self.n_objects = n_objects
        self.__load_dataset__()

    def sub_sampling_function(self, X, Y):
        return sub_sampling_discrete_choices_from_relevance(Xt=X, Yt=Y, n_objects=self.n_objects)

    def splitter(self, iter):
        pass

    def get_train_test_datasets(self, n_datasets):
        pass

    def get_single_train_test_split(self):
        X_train, Y_train = self.sub_sampling_from_dictionary(train_test="train")
        X_test, Y_test = self.sub_sampling_from_dictionary(train_test="test")
        X_train, X_test = standardize_features(X_train, X_test)
        return X_train, Y_train, X_test, Y_test
