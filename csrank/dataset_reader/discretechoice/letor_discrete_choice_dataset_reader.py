import logging

from sklearn.utils import check_random_state

from csrank.constants import DISCRETE_CHOICE
from csrank.util import convert_to_label_encoding
from .util import sub_sampling_discrete_choices
from ..letor_dataset_reader import LetorDatasetReader


class LetorDiscreteChoiceDatasetReader(LetorDatasetReader):
    def __init__(self, random_state=None, n_objects=5, **kwargs):
        super(LetorDiscreteChoiceDatasetReader, self).__init__(learning_problem=DISCRETE_CHOICE, **kwargs)
        self.logger = logging.getLogger(LetorDiscreteChoiceDatasetReader.__name__)
        self.random_state = check_random_state(random_state)
        self.n_objects = n_objects
        self.__load_dataset__()

    def sub_sampling_function(self, n):
        return sub_sampling_discrete_choices(self.X_train[n], self.scores_train[n], n_objects=self.n_objects)

    def convert_output(self, ranking_length):
        self.Y = self.Y.argmin(axis=1)
        self.Y = convert_to_label_encoding(self.Y, ranking_length)

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
