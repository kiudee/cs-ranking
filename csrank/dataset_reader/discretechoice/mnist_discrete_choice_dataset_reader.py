import logging
import os

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state

from csrank.constants import DISCRETE_CHOICE
from .util import convert_to_label_encoding
from ..dataset_reader import DatasetReader


class MNISTDiscreteChoiceDatasetReader(DatasetReader):

    def __init__(self, n_train_instances=10000, n_test_instances=10000, n_objects=10, random_state=None, **kwargs):
        super(MNISTDiscreteChoiceDatasetReader, self).__init__(learning_problem=DISCRETE_CHOICE, dataset_folder='mnist',
                                                               **kwargs)
        self.logger = logging.getLogger(MNISTDiscreteChoiceDatasetReader.__name__)
        self.n_test_instances = n_test_instances
        self.n_train_instances = n_train_instances
        self.n_objects = n_objects
        self.random_state = check_random_state(random_state)
        self.n_features = None
        self.__load_dataset__()

    def __load_dataset__(self):
        x_file = os.path.join(self.dirname, "X_raw_features.npy")
        y_file = os.path.join(self.dirname, "y_labels.npy")
        X_raw = np.load(x_file)
        y_labels = np.load(y_file)
        self.n_features = X_raw.shape[1]
        num_classes = len(np.unique(y_labels))
        n_total = self.n_test_instances + self.n_train_instances
        largest_numbers = self.random_state.randint(1, num_classes, size=n_total)
        self.X = np.empty((n_total, self.n_objects, self.n_features))
        self.Y = np.empty(n_total, dtype=int)
        for i in range(n_total):
            remaining = X_raw[y_labels < largest_numbers[i]]
            largest = X_raw[y_labels == largest_numbers[i]]
            indices = self.random_state.choice(len(remaining), size=self.n_objects, replace=False)
            ind = self.random_state.choice(len(largest), size=1)[0]
            choice = largest[ind]
            self.X[i] = remaining[indices]
            position = self.random_state.choice(self.n_objects, size=1)[0]
            self.X[i][position] = choice
            self.Y[i] = position
        self.Y = convert_to_label_encoding(self.Y, self.n_objects)
        self.__check_dataset_validity__()

    def get_single_train_test_split(self):
        return train_test_split(self.X, self.Y, random_state=self.random_state,
                                test_size=self.n_test_instances)

    def splitter(self, iter):
        pass

    def get_dataset_dictionaries(self, lengths=[5, 6]):
        pass

    def get_train_test_datasets(self, n_datasets=5):
        pass
