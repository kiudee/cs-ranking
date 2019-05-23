import collections
import logging
import os
from abc import ABCMeta

import h5py
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.model_selection import ShuffleSplit

from csrank.dataset_reader.dataset_reader import DatasetReader
from csrank.dataset_reader.util import standardize_features
from csrank.util import print_dictionary


class ExpediaDatasetReader(DatasetReader, metaclass=ABCMeta):
    def __init__(self, fold_id=0, **kwargs):
        super(ExpediaDatasetReader, self).__init__(dataset_folder='expedia', **kwargs)
        self.RAW_DATASET_FILE = os.path.join(self.dirname, 'train.csv')
        self.logger = logging.getLogger(ExpediaDatasetReader.__name__)
        self.fold_id = fold_id
        self.X_train = self.Y_train = self.X_test = self.Y_test = dict()

    def __load_dataset__(self):
        if not os.path.isfile(self.hdf5file_path):
            self.create_choices_dataset()
        self.X_dict, self.Y_dict = self.get_choices_dict()
        self.logger.info("Done loading the dataset")

    def get_choices_dict(self):
        file = h5py.File(self.hdf5file_path, 'r')
        lengths = file["lengths"]
        X = dict()
        Y = dict()
        for ranking_length in np.array(lengths):
            self.X = np.array(file["X_{}".format(ranking_length)])
            self.Y = np.array(file["Y_{}".format(ranking_length)])
            self.__check_dataset_validity__()
            X[ranking_length], Y[ranking_length] = self.X, self.Y
        file.close()
        return X, Y

    def create_choices_dataset(self):
        self.logger.info("Writing in hd5 {}".format(self.hdf5file_path))
        X, scores = self._parse_dataset()
        result, freq = self._build_training_buckets(X, scores)
        h5f = h5py.File(self.hdf5file_path, 'w')
        self.logger.info("Frequencies of rankings: {}".format(print_dictionary(freq)))

        for key, value in result.items():
            x, s = value
            h5f.create_dataset('X_' + str(key), data=x, compression='gzip', compression_opts=9)
            h5f.create_dataset('Y_' + str(key), data=s, compression='gzip', compression_opts=9)
        lengths = np.sort(np.array(list(result.keys())))
        h5f.create_dataset('lengths', data=lengths, compression='gzip', compression_opts=9)
        h5f.close()

    def _build_training_buckets(self, X, Y):
        """Separates object ranking data into buckets of the same ranking size."""
        result = dict()
        frequencies = dict()

        for x, y in zip(X, Y):
            n_objects = len(x)
            if n_objects not in result:
                result[n_objects] = ([], [])
            bucket = result[n_objects]
            bucket[0].append(x)
            bucket[1].append(y)
            if n_objects not in frequencies:
                frequencies[n_objects] = 1
            else:
                frequencies[n_objects] += 1

        # Convert all buckets to numpy arrays:
        for k, v in result.items():
            result[k] = np.array(v[0]), np.array(v[1])
        result = collections.OrderedDict(sorted(result.items()))
        return result, frequencies

    def _filter_dataset(self):
        train = pd.read_csv(self.RAW_DATASET_FILE)
        for col in train.columns.values:
            if "id" in col and col != "srch_id":
                del train[col]
            elif is_numeric_dtype(train[col]):
                arr = np.array(train[col])
                fraction = np.isnan(arr).sum() / len(arr)
                if fraction > 0.0:
                    self.logger.info("########################################################################")
                    self.logger.info("Missing values {}: {}".format(col, np.isnan(arr).sum() / len(arr)))
                    self.logger.info("Min {}: Max {}".format(np.nanmin(arr), np.nanmax(arr)))
                    if fraction > 0.50:
                        del train[col]
                    else:
                        train.loc[train[col].isnull(), col] = np.nanmin(arr) - 1
            else:
                del train[col]
        self.train_df = train

    def sub_sampling_from_dictionary(self):
        X = []
        Y = []
        for n in self.X_train.keys():
            if n == self.n_objects:
                x, y = self.X_train[n], self.Y_train[n]
            elif n > self.n_objects:
                x, y = self.sub_sampling_function(self.X_train[n], self.Y_train[n])
            else:
                x = []
            if len(x) != 0:
                if len(X) == 0:
                    X = np.copy(x)
                    Y = np.copy(y)
                else:
                    X = np.concatenate([X, x], axis=0)
                    Y = np.concatenate([Y, y], axis=0)
                self.logger.info("Sampled instances {} from objects {}".format(x.shape[0], n))
        self.logger.info("Sampled instances {} objects {}".format(X.shape[0], X.shape[1]))
        return X, Y

    def splitter(self, iter):
        for train_idx, test_idx in iter:
            x_train, y_train, x_test, y_test = self.X[train_idx], self.Y[train_idx], self.X[test_idx], self.Y[test_idx]
            x_train, x_test = standardize_features(x_train, x_test)
            yield x_train, y_train, x_test, y_test

    def get_single_train_test_split(self):
        splits = dict()
        cv_iter = ShuffleSplit(n_splits=1, random_state=self.random_state, test_size=0.80)
        for n_obj, arr in self.X_dict.items():
            if arr.shape[0] == 1:
                splits[n_obj] = ([0], [0])
            else:
                splits[n_obj] = list(cv_iter.split(arr))[0]
        self.X_train = dict()
        self.Y_train = dict()
        self.X_test = dict()
        self.Y_test = dict()
        for n_obj, itr in splits.items():
            train_idx, test_idx = itr
            self.X_train[n_obj] = np.copy(self.X_dict[n_obj][train_idx])
            self.X_test[n_obj] = np.copy(self.X_dict[n_obj][test_idx])
            self.Y_train[n_obj] = np.copy(self.Y_dict[n_obj][train_idx])
            self.Y_test[n_obj] = np.copy(self.Y_dict[n_obj][test_idx])
        self.X, self.Y = self.sub_sampling_from_dictionary()
        self.__check_dataset_validity__()
        return self.X, self.Y, self.X_test, self.Y_test

    def sub_sampling_function(self, Xt, Yt):
        raise NotImplemented

    def _parse_dataset(self):
        raise NotImplemented

    def get_dataset_dictionaries(self):
        return self.X_train, self.Y_train, self.X_test, self.Y_test
