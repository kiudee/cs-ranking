import glob
import logging
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import ShuffleSplit
from sklearn.utils import check_random_state

from csrank.constants import LABEL_RANKING
from csrank.numpy_util import ranking_ordering_conversion
from ..dataset_reader import DatasetReader


class IntelligentSystemGroupDatasetReader(DatasetReader):
    def __init__(self, random_state=None, **kwargs):
        super(IntelligentSystemGroupDatasetReader, self).__init__(learning_problem=LABEL_RANKING,
                                                                  dataset_folder='intelligent_system_data', **kwargs)

        self.logger = logging.getLogger(IntelligentSystemGroupDatasetReader.__name__)
        self.train_files = glob.glob(os.path.join(self.dirname, '*.txt'))
        self.train_files.sort()
        self.train_files_names = []
        self.random_state = check_random_state(random_state)
        self.X = dict()
        self.Y = dict()
        self.__load_dataset__()

    def __load_dataset__(self):
        for i, file in enumerate(self.train_files):
            raw_data = pd.read_csv(file, skiprows=[1], delimiter='\t')
            feature_cols = [x for x in raw_data.columns if x[0] == 'A']
            label_cols = [x for x in raw_data.columns if x[0] == 'L' or 'O']
            features = raw_data.loc[:, feature_cols]
            labels = raw_data.loc[:, label_cols]
            if 'O1' in raw_data.columns:
                labels = ranking_ordering_conversion(labels)
            labels -= np.min(labels)
            name = os.path.basename(file).split('.')[0]
            self.train_files_names.append(name)
            self.X[name] = features.as_matrix()
            self.Y[name] = labels.as_matrix()
        self.logger.info("Dataset files: " + repr(self.train_files_names))

    def get_single_train_test_split(self, name="cold"):
        cv_iter = ShuffleSplit(n_splits=1, test_size=0.3, random_state=self.random_state)
        (train_idx, test_idx) = list(cv_iter.split(self.X[name]))[0]
        return self.X[name][train_idx], self.Y[name][train_idx], self.X[name][test_idx], self.Y[name][
            test_idx]

    def get_dataset_dictionaries(self, name="cold"):
        return self.X[name], self.Y[name]
