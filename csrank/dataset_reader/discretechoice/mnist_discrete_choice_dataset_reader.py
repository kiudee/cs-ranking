import numpy as np

from csrank.constants import DISCRETE_CHOICE
from .util import convert_to_label_encoding, angle_between
from ..mnist_dataset_reader import MNISTDatasetReader


class MNISTDiscreteChoiceDatasetReader(MNISTDatasetReader):
    def __init__(self, dataset_type='unique', **kwargs):
        dataset_func_dict = {"unique": self.create_dataset_unique, "largest": self.create_dataset_largest,
                             "median": self.create_dataset_median,
                             'unique_max_occurring': self.create_dataset_mode_least_angle}
        if dataset_type not in dataset_func_dict.keys():
            dataset_type = "median"
        self.dataset_function = dataset_func_dict[dataset_type]
        super(MNISTDiscreteChoiceDatasetReader, self).__init__(learning_problem=DISCRETE_CHOICE, **kwargs)
        self.logger.info("Dataset type {}".format(dataset_type))

    def create_dataset_median(self):
        self.logger.info("create_dataset_median")
        if self.n_objects % 2 == 0:
            self.n_objects = self.n_objects - 1
            self.logger.info(
                "Cannot create the dataset for even numbered sets so decreasing the set size {}".format(self.n_objects))

        n_total = self.n_test_instances + self.n_train_instances
        self.X = np.empty((n_total, self.n_objects, self.n_features))
        self.Y = np.empty(n_total, dtype=int)
        all_indices = np.arange(len(self.X_raw))
        for i in range(n_total):
            choice = np.arange(3)
            while len(choice) != 1:
                indices = self.random_state.choice(all_indices, size=self.n_objects, replace=False)
                labels = self.y_labels[indices]
                choice = np.where(labels == np.median(labels))[0]
                if len(choice) == 1:
                    self.X[i] = self.X_raw[indices]
                    self.Y[i] = choice
        self.Y = convert_to_label_encoding(self.Y, self.n_objects)
        self.__check_dataset_validity__()

    def create_dataset_largest(self):
        self.logger.info("create_dataset_largest new")
        n_total = self.n_test_instances + self.n_train_instances
        self.X = np.empty((n_total, self.n_objects, self.n_features))
        self.Y = np.empty(n_total, dtype=int)
        all_indices = np.arange(len(self.X_raw))
        for i in range(n_total):
            choice = np.arange(3)
            while len(choice) != 1:
                indices = self.random_state.choice(all_indices, size=self.n_objects, replace=False)
                labels = self.y_labels[indices]
                choice = np.where(labels == np.max(labels))[0]
                if len(choice) == 1:
                    self.X[i] = self.X_raw[indices]
                    self.Y[i] = choice
        self.Y = convert_to_label_encoding(self.Y, self.n_objects)
        self.__check_dataset_validity__()

    def create_dataset_mode_least_angle(self):
        self.logger.info("create_dataset_maximum_occurring_number")
        n_total = self.n_test_instances + self.n_train_instances
        self.X = np.empty((n_total, self.n_objects, self.n_features))
        self.Y = np.empty(n_total, dtype=int)
        all_indices = np.arange(len(self.X_raw))
        weights = self.random_state.randn(self.n_features)
        for i in range(n_total):
            while True:
                indices = self.random_state.choice(all_indices, size=self.n_objects, replace=False)
                labels = self.y_labels[indices]
                numbers, counts = np.unique(labels, return_counts=True)
                max_count = np.where(counts == np.max(counts))[0]
                if len(max_count) == 1:
                    self.X[i] = self.X_raw[indices]
                    largest_set = np.where(labels == numbers[max_count])[0]
                    scores = np.array([angle_between(x, weights) for x in self.X[i][largest_set]])
                    self.Y[i] = largest_set[np.argmax(scores)]
                    break
        self.Y = convert_to_label_encoding(self.Y, self.n_objects)
        self.__check_dataset_validity__()

    def create_dataset_unique(self):
        self.logger.info("create_dataset_unique")
        num_classes = len(np.unique(self.y_labels))
        n_total = self.n_test_instances + self.n_train_instances
        unique_number = self.random_state.randint(0, num_classes, size=n_total)
        self.X = np.empty((n_total, self.n_objects, self.n_features))
        self.Y = np.empty(n_total, dtype=int)
        classes = np.arange(num_classes)
        for i in range(n_total):
            s = int((self.n_objects - 1) / 2)
            choices = self.random_state.choice(np.delete(classes, unique_number[i]), size=s)
            ind = self.random_state.choice(np.where(self.y_labels == unique_number[i])[0], size=1)[0]
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
            position = np.where(self.y_labels[indices] == unique_number[i])
            self.Y[i] = position[0][0]
        self.Y = convert_to_label_encoding(self.Y, self.n_objects)
        self.__check_dataset_validity__()
