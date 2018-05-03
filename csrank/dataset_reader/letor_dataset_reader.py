import glob
import logging
import os
from abc import ABCMeta

import numpy as np

from csrank.dataset_reader.dataset_reader import DatasetReader


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
        hdf5_files = glob.glob(os.path.join(self.dirname, self.learning_problem, "*_{}.h5".format(self.year)))
        self.logger.info("For Year {}".format(self.year))

        for file in hdf5_files:
            if not os.path.isfile(file):
                self.__load_dataset__()
            if "train" in os.path.basename(file):
                self.train_file = file
                self.logger.info("Train file {}".format(os.path.basename(file)))
            if "test" in os.path.basename(file):
                self.test_file = file
                self.logger.info("Test file {}".format(os.path.basename(file)))

    def __load_dataset__(self):
        mq_2007_files = glob.glob(os.path.join(self.dirname, self.DATASET_FOLDER_2007, '*.txt'))
        mq_2008_files = glob.glob(os.path.join(self.dirname, self.DATASET_FOLDER_2008, '*.txt'))
        self.logger.info("Query features {}".format(self.query_feature_indices))
        self.logger.info("Query Document features {}".format(self.query_document_feature_indices))
        if self.year == "2007":
            self.dataset, self.test_dataset = self.create_dataset_dictionary(mq_2007_files)
        else:
            self.dataset, self.test_dataset = self.create_dataset_dictionary(mq_2008_files)

    def create_dataset_dictionary(self, files):
        test_dataset = dict()
        dataset = dict()
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
