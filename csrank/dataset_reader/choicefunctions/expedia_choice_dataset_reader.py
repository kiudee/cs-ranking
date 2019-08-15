import logging
import os

from sklearn.utils import check_random_state

from csrank.constants import CHOICE_FUNCTION
from .util import sub_sampling_choices_from_relevance
from ..expedia_dataset_reader import ExpediaDatasetReader


class ExpediaChoiceDatasetReader(ExpediaDatasetReader):
    def __init__(self, random_state=None, n_objects=5, **kwargs):
        super(ExpediaChoiceDatasetReader, self).__init__(learning_problem=CHOICE_FUNCTION, **kwargs)
        self.logger = logging.getLogger(ExpediaChoiceDatasetReader.__name__)
        self.random_state = check_random_state(random_state)
        self.n_objects = n_objects
        self.hdf5file_path = os.path.join(self.dirname, 'exp_choice.h5')
        self.__load_dataset__()

    def sub_sampling_function(self, X, Y):
        return sub_sampling_choices_from_relevance(Xt=X, Yt=Y, n_objects=self.n_objects, offset=1)

    def get_train_test_datasets(self, n_datasets):
        pass

    def _parse_dataset(self):
        self._filter_dataset()
        X = []
        Y = []
        for name, group in self.train_df.groupby("srch_id"):
            del group["srch_id"]
            vals = group.values
            y = vals[:, -1] + vals[:, -2]
            X.append(vals[:, 0:-2])
            y[y != 0] = 1
            Y.append(y)
        return X, Y
