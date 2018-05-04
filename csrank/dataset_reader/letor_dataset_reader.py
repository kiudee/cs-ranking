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
    def __init__(self, year=2007, **kwargs):
        super(LetorDatasetReader, self).__init__(dataset_folder='letor', **kwargs)
        self.DATASET_FOLDER_2007 = 'MQ{}-list'.format(year)
        self.DATASET_FOLDER_2008 = 'MQ{}-list'.format(year)
        self.logger = logging.getLogger('LetorDataset')
        self.year = year
        self.query_feature_indices = [4, 5, 6, 19, 20, 21, 34, 35, 36]
        self.query_document_feature_indices = np.delete(np.arange(0, 46), self.query_feature_indices)

        if (self.year not in [2007, 2008]):
            self.year = 2007

        self.logger.info("For Year {}".format(self.year))
        self.train_file = os.path.join(self.dirname, "train_{}.h5".format(self.year))
        self.test_file = os.path.join(self.dirname, "test_{}.h5".format(self.year))

    def __load_dataset__(self):
        if not (os.path.isfile(self.train_file) and os.path.isfile(self.test_file)):
            self.logger.info("HDF5 datasets not created.....")
            mq_2007_files = glob.glob(os.path.join(self.dirname, self.DATASET_FOLDER_2007, '*.txt'))
            mq_2008_files = glob.glob(os.path.join(self.dirname, self.DATASET_FOLDER_2008, '*.txt'))
            self.logger.info("Query features {}".format(self.query_feature_indices))
            self.logger.info("Query Document features {}".format(self.query_document_feature_indices))
            if self.year == "2007":
                self.dataset, self.test_dataset = self.create_dataset_dictionary(mq_2007_files)
            else:
                self.dataset, self.test_dataset = self.create_dataset_dictionary(mq_2008_files)
            self.create_rankings_dataset(self.dataset, self.train_file)
            self.create_rankings_dataset(self.test_dataset, self.test_file)

    def create_rankings_dataset(self, dataset, hdf5file_path):

        def _build_training_buckets(X, Y):
            """Separates object ranking data into buckets of the same ranking
               size."""
            result = dict()
            freq = dict()

            for x, y in zip(X, Y):
                n_objects = len(x)
                if n_objects not in result:
                    result[n_objects] = ([], [])
                bucket = result[n_objects]
                bucket[0].append(x)
                bucket[1].append(y)
                if n_objects not in freq:
                    freq[n_objects] = 1
                else:
                    freq[n_objects] += 1

            # Convert all buckets to numpy arrays:
            n_instances = sum(freq.values())
            for k, v in result.items():
                result[k] = np.array(v[0]), np.array(v[1])
                freq[k] /= n_instances

            return result, freq

        def reprocess(features, rankings):
            rankings = rankings - 1
            features = np.flip(features, 1)
            rankings = np.flip(rankings, 1)
            for c in rankings:
                np.random.shuffle(c)
            features = np.array([features[i][rankings[i], :] for i in range(rankings.shape[0])])
            return features, rankings

        X = []
        Y = []
        scores = []
        for k, v in dataset.items():
            x = np.array(v)
            X.append(x[:, 0:-1])
            scores.append(x[:, -1])
            Y.append(rankdata(x[:, -1], method='max'))
        X = np.array(X)
        Y = np.array(Y)
        result, freq = _build_training_buckets(X, Y)
        h5f = h5py.File(hdf5file_path, 'w')
        self.logger.info("Frequencies of rankings: {}".format(print_dictionary(freq)))
        self.logger.info("Writing in hd5 {}".format(hdf5file_path))
        for key, value in result.items():
            x, y = value
            x, y = reprocess(x, y)
            h5f.create_dataset('X_' + str(key), data=x, compression='gzip', compression_opts=9)
            h5f.create_dataset('Y_' + str(key), data=y, compression='gzip', compression_opts=9)
            self.logger.info("length {}".format(key))
        lengths = np.array(list(result.keys()))
        h5f.create_dataset('lengths', data=lengths, compression='gzip', compression_opts=9)
        h5f.close()

    def create_dataset_dictionary(self, files):
        test_dataset = dict()
        dataset = dict()
        self.logger.info("Files {}".format(files))
        for file in files:
            for line in open(file):
                information = line.split('#')[0].split(" qid:")
                rel_deg = int(information[0])
                qid = information[1].split(' ')[0]
                x = np.array([float(l.split(':')[1]) for l in information[1].split(' ')[1:-1]])
                x = np.insert(x, len(x), rel_deg)
                if os.path.basename(file) == "I5.txt":
                    if qid not in test_dataset.keys():
                        test_dataset[qid] = [x]
                    else:
                        test_dataset[qid].append(x)
                else:
                    if qid not in dataset.keys():
                        dataset[qid] = [x]
                    else:
                        dataset[qid].append(x)
            if os.path.basename(file) == "I5.txt":
                array = np.array([len(i) for i in test_dataset.values()])
            else:
                array = np.array([len(i) for i in dataset.values()])

            self.logger.info('File name {}'.format(os.path.basename(file)))
            self.logger.info('Maximum length of ranking: {}'.format(np.max(array)))
        return dataset, test_dataset