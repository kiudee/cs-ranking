import collections
import glob
import logging
import os
from abc import ABCMeta

import h5py
import numpy as np

from csrank.dataset_reader.util import standardize_features
from csrank.util import print_dictionary, create_dir_recursively
from .dataset_reader import DatasetReader


class LetorRankingDatasetReader(DatasetReader, metaclass=ABCMeta):
    def __init__(self, year=2007, fold_id=0, exclude_qf=False, **kwargs):
        super(LetorRankingDatasetReader, self).__init__(dataset_folder='letor', **kwargs)
        self.DATASET_FOLDER_2007 = 'MQ{}'.format(year)
        self.DATASET_FOLDER_2008 = 'MQ{}'.format(year)
        self.logger = logging.getLogger(LetorRankingDatasetReader.__name__)

        if year not in [2007, 2008]:
            self.year = 2007
        else:
            self.year = year
        self.exclude_qf = exclude_qf
        self.query_feature_indices = [4, 5, 6, 19, 20, 21, 34, 35, 36]
        self.query_document_feature_indices = np.delete(np.arange(0, 46), self.query_feature_indices)
        self.logger.info("For Year {} excluding query features {}".format(self.year, self.exclude_qf))

        self.dataset_indices = np.arange(5)
        self.condition = np.zeros(5, dtype=bool)
        self.file_format = os.path.join(self.dirname, str(self.year), "S{}.h5")
        create_dir_recursively(self.file_format, is_file_path=True)

        for i in self.dataset_indices:
            h5py_file_pth = self.file_format.format(i + 1)
            self.condition[i] = os.path.isfile(h5py_file_pth)
            if not os.path.isfile(h5py_file_pth):
                self.logger.info("File {} not created".format(h5py_file_pth))
        assert fold_id in self.dataset_indices, "For fold {} no test dataset present".format(fold_id + 1)
        self.logger.info("Test dataset is S{}".format(fold_id + 1))
        self.fold_id = fold_id

    def __load_dataset__(self):
        if not (self.condition.all()):
            self.logger.info("HDF5 datasets not created.....")
            self.logger.info("Query features {}".format(self.query_feature_indices))
            self.logger.info("Query Document features {}".format(self.query_document_feature_indices))
            if self.year == "2007":
                mq_2007_files = glob.glob(os.path.join(self.dirname, self.DATASET_FOLDER_2007, "S*.txt"))
                self.dataset_dictionaries = self.create_dataset_dictionary(mq_2007_files)
            else:
                mq_2008_files = glob.glob(os.path.join(self.dirname, self.DATASET_FOLDER_2008, "S*.txt"))
                self.dataset_dictionaries = self.create_dataset_dictionary(mq_2008_files)
            for key, dataset in self.dataset_dictionaries.items():
                hdf5file_path = self.file_format.format(key.split('S')[-1])
                self.create_choices_dataset(dataset, hdf5file_path)
        self.X_train = dict()
        self.Y_train = dict()
        for i in self.dataset_indices:
            h5py_file_path = self.file_format.format(i + 1)
            if i != self.fold_id:
                X, Y = self.get_choices_dict(h5py_file_path)
                self.merge_to_train(X, Y)
            else:
                self.X_test, self.Y_test = self.get_choices_dict(h5py_file_path)
        self.logger.info("Done loading the dataset")

    def create_choices_dataset(self, dataset, hdf5file_path):
        self.logger.info("Writing in hd5 {}".format(hdf5file_path))
        X, scores = self.create_instances(dataset)
        result, freq = self._build_training_buckets(X, scores)
        h5f = h5py.File(hdf5file_path, 'w')
        self.logger.info("Frequencies of rankings: {}".format(print_dictionary(freq)))

        for key, value in result.items():
            x, s = value
            h5f.create_dataset('X_' + str(key), data=x, compression='gzip', compression_opts=9)
            h5f.create_dataset('score_' + str(key), data=s, compression='gzip', compression_opts=9)
        lengths = np.array(list(result.keys()))
        h5f.create_dataset('lengths', data=lengths, compression='gzip', compression_opts=9)
        h5f.close()

    def get_choices_dict(self, h5py_file_path):
        file = h5py.File(h5py_file_path, 'r')
        lengths = file["lengths"]
        X = dict()
        Y = dict()
        for ranking_length in np.array(lengths):
            self.X = np.array(file["X_{}".format(ranking_length)])
            if self.exclude_qf:
                self.X = self.X[:, :, self.query_document_feature_indices]
            self.Y = np.array(file["score_{}".format(ranking_length)])
            self.Y[self.Y == 2] = 1
            self.__check_dataset_validity__()
            X[ranking_length], Y[ranking_length] = self.X, self.Y
        file.close()
        return X, Y

    def merge_to_train(self, X, Y):
        for key in X.keys():
            x = X[key]
            y = Y[key]
            if key in self.X_train.keys():
                self.X_train[key] = np.append(self.X_train[key], x, axis=0)
                self.Y_train[key] = np.append(self.Y_train[key], y, axis=0)
            else:
                self.X_train[key] = x
                self.Y_train[key] = y

    def sub_sampling_from_dictionary(self, train_test="train"):
        X = []
        Y = []
        if train_test == "train":
            Xt = self.X_train
            Yt = self.Y_train
        elif train_test == "test":
            Xt = self.X_test
            Yt = self.Y_test
        for n in Xt.keys():
            if n >= self.n_objects:
                x, y = self.sub_sampling_function(Xt[n], Yt[n])
                if len(x) != 0:
                    if len(X) == 0:
                        X = np.copy(x)
                        Y = np.copy(y)
                    else:
                        X = np.concatenate([X, x], axis=0)
                        Y = np.concatenate([Y, y], axis=0)
        self.logger.info("Sampled instances {} objects {}".format(X.shape[0], X.shape[1]))
        return X, Y

    def create_instances(self, dataset):
        X = []
        rel_scores = []
        for k, v in dataset.items():
            x = np.array(v)
            s = x[:, -1]
            f = x[:, 0:-1]
            if not np.all(s == 0):
                rel_scores.append(s)
                X.append(f)
        X = np.array(X)
        rel_scores = np.array(rel_scores)
        return X, rel_scores

    def _build_training_buckets(self, X, scores):
        """Separates object ranking data into buckets of the same ranking size."""
        result = dict()
        frequencies = dict()

        for x, s in zip(X, scores):
            n_objects = len(x)
            if n_objects not in result:
                result[n_objects] = ([], [])
            bucket = result[n_objects]
            bucket[0].append(x)
            bucket[1].append(s)
            if n_objects not in frequencies:
                frequencies[n_objects] = 1
            else:
                frequencies[n_objects] += 1
        # Convert all buckets to numpy arrays:
        for k, v in result.items():
            result[k] = (np.array(v[0]), np.array(v[1]))
        result = collections.OrderedDict(sorted(result.items()))
        return result, frequencies

    def create_dataset_dictionary(self, files):
        self.logger.info("Files {}".format(files))
        dataset_dictionaries = dict()
        for file in files:
            dataset = dict()
            key = os.path.basename(file).split('.txt')[0]
            self.logger.info('File name {}'.format(key))
            for line in open(file):
                information = line.split('#')[0].split(" qid:")
                rel_deg = int(information[0])
                qid = information[1].split(' ')[0]
                x = np.array([float(l.split(':')[1]) for l in information[1].split(' ')[1:-1]])
                x = np.insert(x, len(x), rel_deg)
                if qid not in dataset.keys():
                    dataset[qid] = [x]
                else:
                    dataset[qid].append(x)
            array = np.array([len(i) for i in dataset.values()])
            dataset_dictionaries[key] = dataset
            self.logger.info('Maximum length of ranking: {}'.format(np.max(array)))
        return dataset_dictionaries

    def sub_sampling_function(self, Xt, Yt):
        pass

    def get_dataset_dictionaries(self):
        self.X_test, self.X_test = standardize_features(self.X, self.X_test)
        return self.X_train, self.Y_train, self.X_test, self.Y_test

    def get_single_train_test_split(self):
        self.X, self.Y = self.sub_sampling_from_dictionary(train_test="train")
        self.__check_dataset_validity__()
        self.X, self.X_test = standardize_features(self.X, self.X_test)
        return self.X, self.Y, self.X_test, self.Y_test
