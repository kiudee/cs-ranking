import logging
import os

import numpy as np
from sklearn.utils import check_random_state

from csrank.constants import DISCRETE_CHOICE
from .util import sub_sampling_discrete_choices_from_relevance
from ..expedia_dataset_reader import ExpediaDatasetReader


class ExpediaDiscreteChoiceDatasetReader(ExpediaDatasetReader):
    def __init__(self, random_state=None, n_objects=5, **kwargs):
        super(ExpediaDiscreteChoiceDatasetReader, self).__init__(learning_problem=DISCRETE_CHOICE, **kwargs)
        self.logger = logging.getLogger(ExpediaDiscreteChoiceDatasetReader.__name__)
        self.random_state = check_random_state(random_state)
        self.n_objects = n_objects
        self.hdf5file_path = os.path.join(self.dirname, 'exp_dc.h5')
        self.__load_dataset__()

    def sub_sampling_function(self, X, Y):
        return sub_sampling_discrete_choices_from_relevance(Xt=X, Yt=Y, n_objects=self.n_objects)

    def get_train_test_datasets(self, n_datasets):
        pass

    def _parse_dataset(self):
        self._filter_dataset()
        X = []
        Y = []
        l = []
        for name, group in self.train_df.groupby("srch_id"):
            del group["srch_id"]
            vals = group.values
            y = vals[:, -1] + vals[:, -2]
            if (2 in y) or (y.sum() == 1.0):
                X.append(vals[:, 0:-2])
                y[y == np.max(y)] = 1
                y[y != np.max(y)] = 0
                Y.append(y)
            else:
                l.append(y.sum())
        self.logger.info("Percentage of clicked {}".format(len(l) / len(Y)))
        return X, Y
