from sklearn.utils import check_random_state

from csrank.constants import DYAD_RANKING
from csrank.dataset_reader import LetorDatasetReader


class LetorDyadRankingDatasetReader(LetorDatasetReader):
    def __init__(self, random_state=None, **kwargs):
        super(LetorDyadRankingDatasetReader, self).__init__(learning_problem=DYAD_RANKING, **kwargs)
        self.random_state = check_random_state(random_state)

    def splitter(self, iter):
        pass

    def get_train_test_datasets(self, n_datasets=5):
        pass

    def get_dataset_dictionaries(self):
        pass

    def get_single_train_test_split(self):
        pass
