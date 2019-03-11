import logging

from sklearn.utils import check_random_state

from csrank.constants import CHOICE_FUNCTION
from .util import sub_sampling_choices_from_relevance
from ..letor_ranking_dataset_reader import LetorRankingDatasetReader


class LetorRankingChoiceDatasetReader(LetorRankingDatasetReader):
    def __init__(self, random_state=None, n_objects=5, **kwargs):
        super(LetorRankingChoiceDatasetReader, self).__init__(learning_problem=CHOICE_FUNCTION, **kwargs)
        self.logger = logging.getLogger(LetorRankingChoiceDatasetReader.__name__)
        self.random_state = check_random_state(random_state)
        self.n_objects = n_objects
        self.__load_dataset__()

    def sub_sampling_function(self, X, Y):
        return sub_sampling_choices_from_relevance(Xt=X, Yt=Y, n_objects=self.n_objects)

    def splitter(self, iter):
        pass

    def get_train_test_datasets(self, n_datasets):
        pass
