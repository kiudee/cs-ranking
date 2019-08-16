import logging

from sklearn.utils import check_random_state

from csrank.constants import OBJECT_RANKING
from .util import sub_sampling_rankings
from ..letor_listwise_dataset_reader import LetorListwiseDatasetReader


class LetorListwiseObjectRankingDatasetReader(LetorListwiseDatasetReader):
    def __init__(self, random_state=None, n_objects=5, **kwargs):
        super(LetorListwiseObjectRankingDatasetReader, self).__init__(learning_problem=OBJECT_RANKING, **kwargs)
        self.logger = logging.getLogger(LetorListwiseObjectRankingDatasetReader.__name__)
        self.random_state = check_random_state(random_state)
        self.n_objects = n_objects

    def sub_sampling_function(self, X, Y):
        return sub_sampling_rankings(Xt=X, Yt=Y, n_objects=self.n_objects)

    def splitter(self, iter):
        pass

    def get_train_test_datasets(self, n_datasets):
        pass
