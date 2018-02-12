import logging

import h5py
import numpy as np
from sklearn.utils import check_random_state

from csrank.constants import OBJECT_RANKING
from csrank.dataset_reader.letor_dataset_reader import LetorDatasetReader
from csrank.dataset_reader.objectranking.util import sub_sampling

NAME = "LetorObjectRankingDatasetReader"


class LetorObjectRankingDatasetReader(LetorDatasetReader):
    def __init__(self, random_state=None, train_obj=5, **kwargs):
        super(LetorObjectRankingDatasetReader, self).__init__(learning_problem=OBJECT_RANKING, **kwargs)
        self.logger = logging.getLogger(NAME)
        self.random_state = check_random_state(random_state)
        self.train_obj = train_obj
        self.__load_dataset__()

    def __load_dataset__(self):
        file = h5py.File(self.train_file, 'r')
        self.X_train, self.Y_train = self.get_rankings_dict(file)
        if self.train_obj is None:
            self.train_obj = 5
        self.X_train, self.Y_train = self.sub_sampling_for_dictionary()
        file = h5py.File(self.test_file, 'r')
        self.X_test, self.Y_test = self.get_rankings_dict(file)
        self.logger.info("Done loading the dataset")

    def get_rankings_dict(self, file):
        lengths = file["lengths"]
        X = dict()
        Y = dict()
        for ranking_length in np.array(lengths):
            features = np.array(file["X_{}".format(ranking_length)])
            rankings = np.array(file["Y_{}".format(ranking_length)])
            X[ranking_length], Y[ranking_length] = self.X, self.rankings = features, rankings
            self.__check_dataset_validity__()
        return X, Y

    def sub_sampling_for_dictionary(self):
        X = []
        Y = []
        for n in self.X_train.keys():
            if n > self.train_obj:
                x, y = sub_sampling(NAME, self.X_train[n], self.Y_train[n], n_objects=self.train_obj)
                if len(X) == 0:
                    X = np.copy(x)
                    Y = np.copy(y)
                else:
                    X = np.concatenate([X, x], axis=0)
                    Y = np.concatenate([Y, y], axis=0)
        if self.train_obj in self.X_train.keys():
            X = np.concatenate([X, np.copy(self.X_train[self.train_obj])], axis=0)
            Y = np.concatenate([Y, np.copy(self.Y_train[self.train_obj])], axis=0)
        self.logger.info("Sampled instances {} objects {}".format(X.shape[0], X.shape[1]))
        return X, Y

    def splitter(self, iter):
        pass

    def get_train_test_datasets(self, n_datasets):
        return self.X_train, self.Y_train, self.X_test, self.Y_test

    def get_complete_dataset(self):
        pass

    def get_single_train_test_split(self):
        return self.X_train, self.Y_train, self.X_test, self.Y_test

# if __name__ == '__main__':
#     import sys
#     import os
#     import inspect
#     dirname = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
#     logging.basicConfig(filename=os.path.join(dirname, 'log.log'), level=logging.DEBUG,
#                         format='%(asctime)s %(name)s %(levelname)-8s %(message)s',
#                         datefmt='%Y-%m-%d %H:%M:%S')
#     logger = logging.getLogger(name='letor')
#     sys.path.append("..")
#     for n in [2008, 2007]:
#         ds = LetorObjectRankingDatasetReader(year=n)
#         logger.info(ds.X_train.shape)
#         logger.info(np.array(ds.X_test.keys).shape)
