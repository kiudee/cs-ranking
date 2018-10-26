from sklearn.utils import check_random_state

from csrank.constants import DYAD_RANKING
from ..letor_listwise_dataset_reader import LetorListwiseDatasetReader


class LetorDyadRankingListwiseDatasetReader(LetorListwiseDatasetReader):
    def __init__(self, random_state=None, **kwargs):
        super(LetorDyadRankingListwiseDatasetReader, self).__init__(learning_problem=DYAD_RANKING, **kwargs)
        self.random_state = check_random_state(random_state)

    def splitter(self, iter):
        pass

    def get_train_test_datasets(self, n_datasets=5):
        pass

    def get_dataset_dictionaries(self):
        pass

    def get_single_train_test_split(self):
        pass
