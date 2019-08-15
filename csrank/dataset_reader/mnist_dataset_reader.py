import logging
import os

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state

from csrank.dataset_reader.dataset_reader import DatasetReader
from csrank.dataset_reader.util import standardize_features


class MNISTDatasetReader(DatasetReader):
    def __init__(self, n_train_instances=10000, n_test_instances=10000, n_objects=10, random_state=None,
                 standardize=True, **kwargs):
        super(MNISTDatasetReader, self).__init__(dataset_folder='mnist', **kwargs)
        self.logger = logging.getLogger(MNISTDatasetReader.__name__)
        self.n_test_instances = n_test_instances
        self.n_train_instances = n_train_instances
        self.n_objects = n_objects
        self.random_state = check_random_state(random_state)
        self.n_features = None
        self.standardize = standardize
        self.__load_dataset__()
        self.logger.info('Done loading the dataset')

    def __load_dataset__(self):
        x_file = os.path.join(self.dirname, "X_raw_features.npy")
        y_file = os.path.join(self.dirname, "y_labels.npy")
        self.X_raw = np.load(x_file)
        self.y_labels = np.load(y_file)
        self.n_features = self.X_raw.shape[1]

    def get_single_train_test_split(self):
        self.dataset_function()
        x_train, x_test, y_train, y_test = train_test_split(self.X, self.Y, random_state=self.random_state,
                                                            test_size=self.n_test_instances)
        if self.standardize:
            x_train, x_test = standardize_features(x_train, x_test)
        self.logger.info('Done')
        return x_train, y_train, x_test, y_test

    def splitter(self, iter):
        pass

    def get_dataset_dictionaries(self, lengths=[5, 6]):
        x_train = dict()
        y_train = dict()
        x_test = dict()
        y_test = dict()
        for n_obj in lengths:
            self.n_objects = n_obj
            self.dataset_function()
            x_1, x_2, y_1, y_2 = train_test_split(self.X, self.Y, random_state=self.random_state,
                                                  test_size=self.n_test_instances)
            if self.standardize:
                x_1, x_2 = standardize_features(x_1, x_2)
            x_train[n_obj], x_test[n_obj], y_train[n_obj], y_test[n_obj] = x_1, x_2, y_1, y_2
        self.logger.info('Done')
        return x_train, y_train, x_test, y_test

    def get_train_test_datasets(self, n_datasets=5):
        pass
