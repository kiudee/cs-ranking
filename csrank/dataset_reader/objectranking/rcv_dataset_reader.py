import logging
import os

import numpy as np
from joblib import load
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state

from csrank.constants import OBJECT_RANKING
from ..dataset_reader import DatasetReader


class RCVDatasetReader(DatasetReader):

    def __init__(self, n_instances=10000, n_objects=5, query_based=False, random_state=None, **kwargs):
        super(RCVDatasetReader, self).__init__(learning_problem=OBJECT_RANKING, dataset_folder='textual_data', **kwargs)
        self.logger = logging.getLogger(name=RCVDatasetReader.__name__)
        if n_instances not in [10000]:
            raise ValueError('The number of instances should be in %s', str([100, 1000, 10000]))
        if n_objects not in [5, 10]:
            raise ValueError('The number of objects should be in %s', str([5, 10]))

        if (query_based == False):
            file_name = 'rcv1_{}inst_{}obj.pkl'.format(n_instances, n_objects)
        else:
            file_name = 'rcv1_query-based_{}inst_{}obj.pkl'.format(n_instances, n_objects)

        self.train_file = os.path.join(self.dirname, file_name)
        self.logger.info(self.train_file)

        self.random_state = check_random_state(random_state)
        self.__load_dataset__()

    def __load_dataset__(self):
        with open(self.train_file, 'rb') as file:
            data = load(file)
        orderings = data['orderings'].values
        object_indices = (data['orderings']['object_index']).values
        features = (data['features']).values
        X = []
        for indices in object_indices:
            X.append(features[indices])
        self.X = np.array(X)
        self.Y = np.array(orderings)
        for i, x in enumerate(self.X):
            x = StandardScaler().fit_transform(x)
            self.X[i] = x
        self.__check_dataset_validity__()

    def get_train_test_dataset(self):
        cv_iter = ShuffleSplit(n_splits=1, test_size=0.3, random_state=self.random_state)
        splits = list(cv_iter.split(self.X))
        return self.splitter(splits)

    def get_dataset_dictionaries(self):
        return self.X, self.Y

    def splitter(self, iter):
        for train_idx, test_idx in iter:
            yield self.X[train_idx], self.Y[train_idx], self.X[test_idx], self.Y[test_idx]

    def get_train_test_datasets(self, n_datasets):
        pass

    def get_single_train_test_split(self):
        return self.X, self.Y
