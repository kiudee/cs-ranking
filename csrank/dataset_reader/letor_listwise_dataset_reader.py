from abc import ABCMeta
import collections
import glob
import logging
import os

import numpy as np
from scipy.stats import rankdata

from csrank.constants import DISCRETE_CHOICE
from csrank.constants import OBJECT_RANKING
from csrank.dataset_reader.util import standardize_features
from csrank.util import create_dir_recursively
from csrank.util import print_dictionary
from .dataset_reader import DatasetReader

try:
    import h5py
except ImportError:
    from csrank.util import MissingExtraError

    raise MissingExtraError("h5py", "data")
logger = logging.getLogger("asdf")


class LetorListwiseDatasetReader(DatasetReader, metaclass=ABCMeta):
    def __init__(self, year=2007, fold_id=0, exclude_qf=False, **kwargs):
        super(LetorListwiseDatasetReader, self).__init__(
            dataset_folder="letor", **kwargs
        )
        self.DATASET_FOLDER_2007 = "MQ{}-list".format(year)
        self.DATASET_FOLDER_2008 = "MQ{}-list".format(year)

        if year not in {2007, 2008}:
            raise ValueError("year must be either 2007 or 2008")
        self.year = year
        self.exclude_qf = exclude_qf
        self.query_feature_indices = [4, 5, 6, 19, 20, 21, 34, 35, 36]
        self.query_document_feature_indices = np.delete(
            np.arange(0, 46), self.query_feature_indices
        )
        logger.info(
            "For Year {} excluding query features {}".format(self.year, self.exclude_qf)
        )

        self.dataset_indices = np.arange(5)
        self.condition = np.zeros(5, dtype=bool)
        self.file_format = os.path.join(self.dirname, str(self.year), "I{}.h5")
        create_dir_recursively(self.file_format, is_file_path=True)

        for i in self.dataset_indices:
            h5py_file_pth = self.file_format.format(i + 1)
            self.condition[i] = os.path.isfile(h5py_file_pth)
            if not os.path.isfile(h5py_file_pth):
                logger.info("File {} not created".format(h5py_file_pth))
        assert (
            fold_id in self.dataset_indices
        ), "For fold {} no test dataset present".format(fold_id + 1)
        logger.info("Test dataset is I{}".format(fold_id + 1))
        self.fold_id = fold_id
        self.__load_dataset__()

    def __load_dataset__(self):
        if not (self.condition.all()):
            logger.info("HDF5 datasets not created.....")
            logger.info("Query features {}".format(self.query_feature_indices))
            logger.info(
                "Query Document features {}".format(self.query_document_feature_indices)
            )
            if self.year == "2007":
                mq_2007_files = glob.glob(
                    os.path.join(self.dirname, self.DATASET_FOLDER_2007, "I*.txt")
                )
                self.dataset_dictionaries = self.create_dataset_dictionary(
                    mq_2007_files
                )
            else:
                mq_2008_files = glob.glob(
                    os.path.join(self.dirname, self.DATASET_FOLDER_2008, "I*.txt")
                )
                self.dataset_dictionaries = self.create_dataset_dictionary(
                    mq_2008_files
                )
            for key, dataset in self.dataset_dictionaries.items():
                hdf5file_path = self.file_format.format(key.split("I")[-1])
                self.create_rankings_dataset(dataset, hdf5file_path)
        self.X_train = dict()
        self.Y_train = dict()
        self.scores_train = dict()
        for i in self.dataset_indices:
            h5py_file_path = self.file_format.format(i + 1)
            if i != self.fold_id:
                X, Y, S = self.get_rankings_dict(h5py_file_path)
                self.merge_to_train(X, Y, S)
            else:
                self.X_test, self.Y_test, self.scores_test = self.get_rankings_dict(
                    h5py_file_path
                )
        logger.info("Done loading the dataset")

    def create_rankings_dataset(self, dataset, hdf5file_path):
        logger.info("Writing in hd5 {}".format(hdf5file_path))
        X, Y, scores = self.create_instances(dataset)
        result, freq = self._build_training_buckets(X, Y, scores)
        h5f = h5py.File(hdf5file_path, "w")
        logger.info("Frequencies of rankings: {}".format(print_dictionary(freq)))

        for key, value in result.items():
            x, y, s = value
            h5f.create_dataset(
                "X_" + str(key), data=x, compression="gzip", compression_opts=9
            )
            h5f.create_dataset(
                "Y_" + str(key), data=y, compression="gzip", compression_opts=9
            )
            h5f.create_dataset(
                "score_" + str(key), data=s, compression="gzip", compression_opts=9
            )
        lengths = np.array(list(result.keys()))
        h5f.create_dataset(
            "lengths", data=lengths, compression="gzip", compression_opts=9
        )
        h5f.close()

    # def reprocess(self, features, rankings, scores):
    #     features = np.flip(features, 1)
    #     rankings = np.flip(rankings, 1)
    #     for i, r in enumerate(rankings):
    #         np.random.shuffle(r)
    #         scores[i] = scores[i][r]
    #     features = np.array([features[i][rankings[i], :] for i in range(rankings.shape[0])])
    #     return features, rankings, scores

    def create_instances(self, dataset):
        X = []
        Y = []
        scores = []
        for k, v in dataset.items():
            x = np.array(v)
            s = x[:, -1]
            r = (len(s) - rankdata(s, method="average")).astype(int)
            f = x[:, 0:-1]
            indices = np.arange(len(r))
            np.random.shuffle(indices)
            r = r[indices]
            s = s[indices]
            f = f[indices, :]
            scores.append(s)
            X.append(f)
            Y.append(r)
        X = np.array(X)
        Y = np.array(Y)
        scores = np.array(scores)
        return X, Y, scores

    def _build_training_buckets(self, X, Y, scores):
        """Separates object ranking data into buckets of the same ranking size."""
        result = dict()
        frequencies = dict()

        for x, y, s in zip(X, Y, scores):
            n_objects = len(x)
            if n_objects not in result:
                result[n_objects] = ([], [], [])
            bucket = result[n_objects]
            bucket[0].append(x)
            bucket[1].append(y)
            bucket[2].append(s)
            if n_objects not in frequencies:
                frequencies[n_objects] = 1
            else:
                frequencies[n_objects] += 1

        # Convert all buckets to numpy arrays:
        for k, v in result.items():
            result[k] = np.array(v[0]), np.array(v[1]), np.array(v[2])
        result = collections.OrderedDict(sorted(result.items()))
        return result, frequencies

    def create_dataset_dictionary(self, files):
        logger.info("Files {}".format(files))
        dataset_dictionaries = dict()
        for file in files:
            dataset = dict()
            key = os.path.basename(file).split(".txt")[0]
            logger.info("File name {}".format(key))
            for line in open(file):
                information = line.split("#")[0].split(" qid:")
                rel_deg = int(information[0])
                qid = information[1].split(" ")[0]
                x = np.array(
                    [
                        float(elem.split(":")[1])
                        for elem in information[1].split(" ")[1:-1]
                    ]
                )
                x = np.insert(x, len(x), rel_deg)
                if qid not in dataset:
                    dataset[qid] = [x]
                else:
                    dataset[qid].append(x)
            array = np.array([len(i) for i in dataset.values()])
            dataset_dictionaries[key] = dataset
            logger.info("Maximum length of ranking: {}".format(np.max(array)))
        return dataset_dictionaries

    def get_rankings_dict(self, h5py_file_path):
        file = h5py.File(h5py_file_path, "r")
        lengths = file["lengths"]
        X = dict()
        Y = dict()
        scores = dict()
        for ranking_length in np.array(lengths):
            self.X = np.array(file["X_{}".format(ranking_length)])
            if self.exclude_qf:
                self.X = self.X[:, :, self.query_document_feature_indices]
            self.Y = np.array(file["Y_{}".format(ranking_length)])
            self.convert_output(ranking_length)
            s = np.array(file["score_{}".format(ranking_length)])
            self.__check_dataset_validity__()
            X[ranking_length], Y[ranking_length], scores[ranking_length] = (
                self.X,
                self.Y,
                s,
            )
        file.close()
        return X, Y, scores

    def merge_to_train(self, X, Y, scores):
        for key in X.keys():
            x = X[key]
            y = Y[key]
            s = scores[key]
            if key in self.X_train:
                self.X_train[key] = np.append(self.X_train[key], x, axis=0)
                self.Y_train[key] = np.append(self.Y_train[key], y, axis=0)
                self.scores_train[key] = np.append(self.scores_train[key], s, axis=0)
            else:
                self.X_train[key] = x
                self.Y_train[key] = y
                self.scores_train[key] = s

    def sub_sampling_from_dictionary(self, train_test="train"):
        X = []
        Y = []

        if train_test == "train":
            Xt = self.X_train
            Yt = self.Y_train
            St = self.scores_train
        elif train_test == "test":
            Xt = self.X_test
            Yt = self.Y_test
            St = self.scores_test
        for n in Xt.keys():
            if n >= self.n_objects:
                if self.learning_problem == OBJECT_RANKING:
                    x, y = self.sub_sampling_function(Xt[n], Yt[n])
                if self.learning_problem == DISCRETE_CHOICE:
                    x, y = self.sub_sampling_function(Xt[n], St[n])
                if len(x) != 0:
                    if len(X) == 0:
                        X = np.copy(x)
                        Y = np.copy(y)
                    else:
                        X = np.concatenate([X, x], axis=0)
                        Y = np.concatenate([Y, y], axis=0)
        logger.info("Sampled instances {} objects {}".format(X.shape[0], X.shape[1]))
        return X, Y

    def sub_sampling_function(self, Xt, Yt):
        pass

    def convert_output(self, ranking_length):
        pass

    def get_dataset_dictionaries(self):
        self.X_test, self.X_test = standardize_features(self.X, self.X_test)
        return self.X_train, self.Y_train, self.X_test, self.Y_test

    def get_single_train_test_split(self):
        self.X, self.Y = self.sub_sampling_from_dictionary(train_test="train")
        self.X, self.X_test = standardize_features(self.X, self.X_test)
        self.__check_dataset_validity__()
        return self.X, self.Y, self.X_test, self.Y_test
