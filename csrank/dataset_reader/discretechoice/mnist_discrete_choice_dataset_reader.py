import numpy as np

from csrank.constants import DISCRETE_CHOICE
from .util import convert_to_label_encoding
from ..mnist_dataset_reader import MNISTDatasetReader


class MNISTDiscreteChoiceDatasetReader(MNISTDatasetReader):
    def __init__(self, **kwargs):
        super(MNISTDiscreteChoiceDatasetReader, self).__init__(learning_problem=DISCRETE_CHOICE, **kwargs)

    def create_dataset_largest(self):
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
        self.__check_dataset_validity__()

    def create_dataset(self):
        num_classes = len(np.unique(self.y_labels))
        n_total = self.n_test_instances + self.n_train_instances
        largest_numbers = self.random_state.randint(0, num_classes, size=n_total)
        self.X = np.empty((n_total, self.n_objects, self.n_features))
        self.Y = np.empty(n_total, dtype=int)
        classes = np.arange(num_classes)
        for i in range(n_total):
            s = int((self.n_objects - 1) / 2)
            choices = self.random_state.choice(np.delete(classes, largest_numbers[i]), size=s)
            ind = self.random_state.choice(np.where(self.y_labels == largest_numbers[i])[0], size=1)[0]
            indices = [ind]
            for j, c in enumerate(choices):
                remaining = np.where(self.y_labels == c)[0]
                size = 2
                if self.n_objects % 2 == 0 and j == len(choices) - 1:
                    size = 3
                ind = list(self.random_state.choice(remaining, size=size, replace=False))
                indices = indices + ind
            self.random_state.shuffle(indices)
            self.X[i] = self.X_raw[indices]
            position = np.where(self.y_labels[indices] == largest_numbers[i])
            self.Y[i] = position[0][0]
        self.Y = convert_to_label_encoding(self.Y, self.n_objects)
        self.__check_dataset_validity__()
