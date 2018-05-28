import logging

from sklearn.utils import check_random_state

from csrank.constants import OBJECT_RANKING
from .util import sub_sampling_rankings
from ..letor_dataset_reader import LetorDatasetReader


class LetorObjectRankingDatasetReader(LetorDatasetReader):
    def __init__(self, random_state=None, n_objects=5, **kwargs):
        super(LetorObjectRankingDatasetReader, self).__init__(learning_problem=OBJECT_RANKING, **kwargs)
        self.logger = logging.getLogger(LetorObjectRankingDatasetReader.__name__)
        self.random_state = check_random_state(random_state)
        self.n_objects = n_objects
        self.__load_dataset__()

    def sub_sampling_function(self, n):
        return sub_sampling_rankings(self.X_train[n], self.Y_train[n], n_objects=self.n_objects)

    def splitter(self, iter):
        pass

    def get_train_test_datasets(self, n_datasets):
        pass

    def get_dataset_dictionaries(self):
        return self.X_train, self.Y_train, self.X_test, self.Y_test

    def get_single_train_test_split(self):
        self.X, self.Y = self.sub_sampling_from_dictionary()
        self.__check_dataset_validity__()
        return self.X, self.Y, self.X_test, self.Y_test
