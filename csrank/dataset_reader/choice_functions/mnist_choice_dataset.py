import logging
import os

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state

from csrank.constants import CHOICE_FUNCTIONS
from csrank.dataset_reader import DatasetReader


class MNISTChoiceDatasetReader(DatasetReader):

    def __init__(self, n_train_instances=10000, n_test_instances=10000, n_objects=5, random_state=None, **kwargs):
        super(MNISTChoiceDatasetReader, self).__init__(learning_problem=CHOICE_FUNCTIONS, dataset_folder='mnist',
            **kwargs)
        self.logger = logging.getLogger(MNISTChoiceDatasetReader.__name__)
        self.n_test_instances = n_test_instances
        self.n_train_instances = n_train_instances
        self.n_objects = n_objects
        self.random_state = check_random_state(random_state)
        self.n_features = None

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
        y_number = np.empty((n_total, self.n_objects), dtype=int)
        for i in range(n_total):
            remaining = X_raw[y_labels <= largest_numbers[i]]
            while True:
                indices = self.random_state.choice(len(remaining), size=self.n_objects, replace=False)
                self.X[i] = remaining[indices]
                y_number[i] = y_labels[y_labels <= largest_numbers[i]][indices]
                if largest_numbers[i] in y_number[i]:
                    break
        self.Y = (y_number == largest_numbers[:, None]).astype(int)
        self.__check_dataset_validity__()

    def get_single_train_test_split(self):
        return train_test_split(self.X, self.Y, random_state=self.random_state,
            test_size=self.n_test_instances)
