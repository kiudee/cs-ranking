import numpy as np

from csrank.constants import CHOICE_FUNCTION
from ..mnist_dataset_reader import MNISTDatasetReader


class MNISTChoiceDatasetReader(MNISTDatasetReader):
    def __init__(self, dataset_type='unique', **kwargs):
        dataset_func_dict = {"unique": self.create_dataset_unique, "largest": self.create_dataset_largest,
                             'mode': self.create_dataset_mode}
        if dataset_type not in dataset_func_dict.keys():
            dataset_type = "unique"
        self.dataset_function = dataset_func_dict[dataset_type]
        super(MNISTChoiceDatasetReader, self).__init__(learning_problem=CHOICE_FUNCTION, **kwargs)
        self.logger.info("Dataset type {}".format(dataset_type))

    def create_dataset_largest(self):
        self.logger.info("Largest Dataset")
        num_classes = len(np.unique(self.y_labels))
        n_total = self.n_test_instances + self.n_train_instances
        largest_numbers = self.random_state.randint(1, num_classes, size=n_total)
        self.X = np.empty((n_total, self.n_objects, self.n_features))
        y_number = np.empty((n_total, self.n_objects), dtype=int)
        for i in range(n_total):
            remaining = self.X_raw[self.y_labels <= largest_numbers[i]]
            while True:
                indices = self.random_state.choice(len(remaining), size=self.n_objects, replace=False)
                self.X[i] = remaining[indices]
                y_number[i] = self.y_labels[self.y_labels <= largest_numbers[i]][indices]
                if largest_numbers[i] in y_number[i]:
                    break
        self.Y = (y_number == largest_numbers[:, None]).astype(int)
        self.__check_dataset_validity__()

    def create_dataset_mode(self):
        self.logger.info("Mode Dataset")
        n_total = self.n_test_instances + self.n_train_instances
        self.X = np.empty((n_total, self.n_objects, self.n_features))
        self.Y = np.zeros((n_total, self.n_objects), dtype=int)
        all_indices = np.arange(len(self.X_raw))
        for i in range(n_total):
            indices = self.random_state.choice(all_indices, size=self.n_objects, replace=False)
            labels = self.y_labels[indices]
            numbers, counts = np.unique(labels, return_counts=True)
            modes = numbers[np.where(counts == np.max(counts))[0]]
            self.X[i] = self.X_raw[indices]
            self.Y[i] = np.array(np.isin(labels, modes), dtype=int)
        self.__check_dataset_validity__()

    def create_dataset_unique(self):
        self.logger.info("Unique Dataset")
        n_total = self.n_test_instances + self.n_train_instances
        self.X = np.empty((n_total, self.n_objects, self.n_features))
        self.Y = np.zeros((n_total, self.n_objects), dtype=int)
        all_indices = np.arange(len(self.X_raw))
        for i in range(n_total):
            while True:
                indices = self.random_state.choice(all_indices, size=self.n_objects, replace=False)
                labels = self.y_labels[indices]
                numbers, counts = np.unique(labels, return_counts=True)
                unique_numbers = numbers[np.where(counts == 1)[0]]
                if len(unique_numbers) > 1:
                    self.random_state.shuffle(indices)
                    self.X[i] = self.X_raw[indices]
                    labels = self.y_labels[indices]
                    self.Y[i] = np.array(np.isin(labels, unique_numbers), dtype=int)
                    break
        self.__check_dataset_validity__()
