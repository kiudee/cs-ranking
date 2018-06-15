import numpy as np
from csrank.constants import DISCRETE_CHOICE

from .util import convert_to_label_encoding
from ..mnist_dataset_reader import MNISTDatasetReader
from ..util import standardize_features


class MNISTDiscreteChoiceDatasetReader(MNISTDatasetReader):
    def __init__(self, **kwargs):
        super(MNISTDiscreteChoiceDatasetReader, self).__init__(learning_problem=DISCRETE_CHOICE, **kwargs)

    def create_dataset(self):
        num_classes = len(np.unique(self.y_labels))
        n_total = self.n_test_instances + self.n_train_instances
        largest_numbers = self.random_state.randint(1, num_classes, size=n_total)
        self.X = np.empty((n_total, self.n_objects, self.n_features))
        self.Y = np.empty(n_total, dtype=int)
        for i in range(n_total):
            remaining = self.X_raw[self.y_labels < largest_numbers[i]]
            largest = self.X_raw[self.y_labels == largest_numbers[i]]
            indices = self.random_state.choice(len(remaining), size=self.n_objects, replace=False)
            ind = self.random_state.choice(len(largest), size=1)[0]
            choice = largest[ind]
            self.X[i] = remaining[indices]
            position = self.random_state.choice(self.n_objects, size=1)[0]
            self.X[i][position] = choice
            self.Y[i] = position
        self.Y = convert_to_label_encoding(self.Y, self.n_objects)
        self.X = standardize_features(self.X)
        self.__check_dataset_validity__()
