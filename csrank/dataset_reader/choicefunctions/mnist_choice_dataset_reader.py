import numpy as np

from csrank.constants import CHOICE_FUNCTIONS
from csrank.dataset_reader.util import standardize_features
from ..mnist_dataset_reader import MNISTDatasetReader


class MNISTChoiceDatasetReader(MNISTDatasetReader):
    def __init__(self, **kwargs):
        super(MNISTChoiceDatasetReader, self).__init__(learning_problem=CHOICE_FUNCTIONS, **kwargs)

    def create_dataset(self):
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
        self.X = standardize_features(self.X)
        self.__check_dataset_validity__()
