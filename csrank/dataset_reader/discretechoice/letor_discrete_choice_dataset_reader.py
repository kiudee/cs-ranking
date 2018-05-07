import logging

import h5py
from sklearn.utils import check_random_state

from csrank.constants import DISCRETE_CHOICE
from csrank.dataset_reader import LetorDatasetReader
import numpy as np

from csrank.dataset_reader.discretechoice.util import sub_sampling_choices


class LetorDiscreteChoiceDatasetReader(LetorDatasetReader):
    def __init__(self, random_state=None, n_objects=5, **kwargs):
        super(LetorDiscreteChoiceDatasetReader, self).__init__(learning_problem=DISCRETE_CHOICE, **kwargs)
        self.logger = logging.getLogger(LetorDiscreteChoiceDatasetReader.__name__)
        self.random_state = check_random_state(random_state)
        self.n_objects = n_objects
        self.__load_dataset__()

    def __load_dataset__(self):
        super().__load_dataset__()
        file = h5py.File(self.train_file, 'r')
        self.X_train, self.Y_train, self.scores_train = self.get_rankings_dict(file)
        file = h5py.File(self.test_file, 'r')
        self.X_test, self.Y_test, self.scores_test = self.get_rankings_dict(file)
        self.logger.info("Done loading the dataset")

    def get_rankings_dict(self, file):
        lengths = file["lengths"]
        X = dict()
        Y = dict()
        scores = dict()
        for ranking_length in np.array(lengths):
            self.X = np.array(file["X_{}".format(ranking_length)])
            self.Y = np.array(file["Y_{}".format(ranking_length)]).argmin(axis=1)
            s = np.array(file["score_{}".format(ranking_length)])
            self.__check_dataset_validity__()
            X[ranking_length], Y[ranking_length], scores[ranking_length] = self.X, self.Y, s
        return X, Y, scores

    def sub_sampling_for_dictionary(self):
        X = []
        Y = []
        for n in self.X_train.keys():
            if n > self.n_objects:
                x, y = sub_sampling_choices(LetorDiscreteChoiceDatasetReader.__name__, self.X_train[n], self.scores_train[n],
                                             n_objects=self.n_objects)
                if len(X) == 0:
                    X = np.copy(x)
                    Y = np.copy(y)
                else:
                    X = np.concatenate([X, x], axis=0)
                    Y = np.concatenate([Y, y], axis=0)
        if self.n_objects in self.X_train.keys():
            X = np.concatenate([X, np.copy(self.X_train[self.n_objects])], axis=0)
            Y = np.concatenate([Y, np.copy(self.Y_train[self.n_objects])], axis=0)
        self.logger.info("Sampled instances {} objects {}".format(X.shape[0], X.shape[1]))
        return X, Y

    def splitter(self, iter):
        pass

    def get_train_test_datasets(self, n_datasets):
        pass

    def get_dataset_dictionaries(self):
        return self.X_train, self.Y_train, self.X_test, self.Y_test

    def get_single_train_test_split(self):
        self.X, self.Y = self.sub_sampling_for_dictionary()
        self.__check_dataset_validity__()
        return self.X, self.Y, self.X_test, self.Y_test