from sklearn.model_selection import StratifiedKFold, ShuffleSplit
from sklearn.utils import check_random_state

from csrank.constants import DYAD_RANKING
from ..sushi_dataset_reader import SushiDatasetReader


class SushiDyadRankingDatasetReader(SushiDatasetReader):
    def __init__(self, random_state=None, **kwargs):
        super(SushiDyadRankingDatasetReader, self).__init__(learning_problem=DYAD_RANKING, **kwargs)
        self.random_state = check_random_state(random_state)

    def splitter(self, iter):
        for train_idx, test_idx in iter:
            yield self.X[train_idx], self.Xc[train_idx], self.Y[train_idx], self.X[test_idx], self.Xc[
                train_idx], \
                  self.Y[test_idx]

    def get_train_test_datasets(self, n_datasets=5):
        cv_iter = StratifiedKFold(n_splits=n_datasets, shuffle=True, random_state=self.random_state)
        splits = list(cv_iter.split(self.X))
        return self.splitter(splits)

    def get_dataset_dictionaries(self):
        return self.Xc, self.X, self.Y

    def get_single_train_test_split(self):
        cv_iter = ShuffleSplit(n_splits=1, random_state=self.random_state, test_size=0.30)
        splits = list(cv_iter.split(self.X))
        return list(self.splitter(splits))[0]
