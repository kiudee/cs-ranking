import glob
import logging
import os
from abc import ABCMeta

import h5py
import numpy as np
from scipy.stats import rankdata

from csrank.dataset_reader.dataset_reader import DatasetReader
from csrank.util import print_dictionary


class LetorDatasetReader(DatasetReader, metaclass=ABCMeta):
    def __init__(self, year=2007, fold=1, **kwargs):
        super(LetorDatasetReader, self).__init__(dataset_folder='letor', **kwargs)
        self.DATASET_FOLDER_2007 = 'MQ{}-list'.format(year)
        self.DATASET_FOLDER_2008 = 'MQ{}-list'.format(year)
        self.logger = logging.getLogger(LetorDatasetReader.__name__)

        if year not in [2007, 2008]:
            self.year = 2007
        else:
            self.year = year
        self.query_feature_indices = [4, 5, 6, 19, 20, 21, 34, 35, 36]
        self.query_document_feature_indices = np.delete(np.arange(0, 46), self.query_feature_indices)
        self.logger.info("For Year {}".format(self.year))

        self.dataset_indices = np.arange(5) + 1
        self.condition = np.zeros(5, dtype=bool)
        self.file_format = os.path.join(self.dirname, str(self.year), "I{}.h5")

        for i in self.dataset_indices:
            self.condition[i - 1] = os.path.isfile(self.file_format.format(i))
            if not os.path.isfile(self.file_format.format(i)):
                self.logger.info("File {} not created".format(self.file_format.format(i)))
        assert fold in self.dataset_indices, "For fold {} no test dataset present".format(fold)
        self.logger.info("Test dataset is I{}".format(fold))
        self.fold = fold

    def __load_dataset__(self):
        if not (self.condition.all()):
            self.logger.info("HDF5 datasets not created.....")
            self.logger.info("Query features {}".format(self.query_feature_indices))
            self.logger.info("Query Document features {}".format(self.query_document_feature_indices))
            if self.year == "2007":
                mq_2007_files = glob.glob(os.path.join(self.dirname, self.DATASET_FOLDER_2007, '*.txt'))
                self.dataset_dictionaries = self.create_dataset_dictionary(mq_2007_files)
            else:
                mq_2008_files = glob.glob(os.path.join(self.dirname, self.DATASET_FOLDER_2008, '*.txt'))
                self.dataset_dictionaries = self.create_dataset_dictionary(mq_2008_files)
            for key, dataset in self.dataset_dictionaries.items():
                hdf5file_path = self.file_format.format(key.split('I')[-1])
                self.create_rankings_dataset(dataset, hdf5file_path)

    def create_rankings_dataset(self, dataset, hdf5file_path):
        self.logger.info("Writing in hd5 {}".format(hdf5file_path))
        X, Y, scores = self.create_instances(dataset)
        result, freq = self._build_training_buckets(X, Y, scores)
        h5f = h5py.File(hdf5file_path, 'w')
        #self.logger.info("Frequencies of rankings: {}".format(print_dictionary(freq)))
        for key, value in result.items():
            x, y, s = value
            h5f.create_dataset('X_' + str(key), data=x, compression='gzip', compression_opts=9)
            h5f.create_dataset('Y_' + str(key), data=y, compression='gzip', compression_opts=9)
            h5f.create_dataset('score_' + str(key), data=s, compression='gzip', compression_opts=9)
            self.logger.info("length {}".format(key))
        lengths = np.array(list(result.keys()))
        h5f.create_dataset('lengths', data=lengths, compression='gzip', compression_opts=9)
        h5f.close()

    # def reprocess(self, features, rankings, scores):
    #     features = np.flip(features, 1)
    #     rankings = np.flip(rankings, 1)
    #     for i, r in enumerate(rankings):
    #         np.random.shuffle(r)
    #         scores[i] = scores[i][r]
    #     features = np.array([features[i][rankings[i], :] for i in range(rankings.shape[0])])
    #     return features, rankings, scores

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
        n_instances = sum(frequencies.values())
        for k, v in result.items():
            result[k] = np.array(v[0]), np.array(v[1]), np.array(v[2])
            frequencies[k] /= n_instances

        return result, frequencies

    def create_instances(self, dataset):
        X = []
        Y = []
        scores = []
        for k, v in dataset.items():
            x = np.array(v)
            s = x[:, -1]
            r = (len(s) - rankdata(s, method='average')).astype(int)
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
