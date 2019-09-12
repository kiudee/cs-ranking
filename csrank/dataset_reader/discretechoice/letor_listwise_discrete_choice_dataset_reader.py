import logging

from sklearn.utils import check_random_state

from csrank.constants import DISCRETE_CHOICE
from .util import sub_sampling_discrete_choices_from_relevance, convert_to_label_encoding
from ..letor_listwise_dataset_reader import LetorListwiseDatasetReader


class LetorListwiseDiscreteChoiceDatasetReader(LetorListwiseDatasetReader):
    def __init__(self, random_state=None, n_objects=5, **kwargs):
        super(LetorListwiseDiscreteChoiceDatasetReader, self).__init__(learning_problem=DISCRETE_CHOICE, **kwargs)
        self.logger = logging.getLogger(LetorListwiseDiscreteChoiceDatasetReader.__name__)
        self.random_state = check_random_state(random_state)
        self.n_objects = n_objects

    def sub_sampling_function(self, X, Y):
        return sub_sampling_discrete_choices_from_relevance(Xt=X, Yt=Y, n_objects=self.n_objects)

    def convert_output(self, ranking_length):
        self.Y = self.Y.argmin(axis=1)
        self.Y = convert_to_label_encoding(self.Y, ranking_length)

    def splitter(self, iter):
        pass

    def get_train_test_datasets(self, n_datasets):
        pass
