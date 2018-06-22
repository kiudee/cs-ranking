import os

import numpy as np
import pandas as pd
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.utils import check_random_state

from csrank.constants import LABEL_RANKING
from csrank.numpy_util import ranking_ordering_conversion
from ..dataset_reader import DatasetReader


class SurveyDatasetReader(DatasetReader):
    def __init__(self, random_state=None, **kwargs):
        super(SurveyDatasetReader, self).__init__(learning_problem=LABEL_RANKING, dataset_folder='survey_data',
                                                  **kwargs)
        self.train_file = os.path.join(self.dirname, 'rawdata_all.dta')
        self.random_state = check_random_state(random_state)
        self.__load_dataset__()

    def __load_dataset__(self):
        df = pd.io.stata.read_stata(self.train_file)
        orderings = []
        features = []
        for row in df.itertuples():
            orderings.append(row[4:8])
            context_feature = [float(i) if i != '.' else np.NAN for i in row[13:33]]
            features.append(context_feature)
        X = np.array(features)
        X = Imputer().fit_transform(X)
        X = np.array([np.log(np.array(X[:, i]) + 1) for i in range(len(features[0]))])
        X = np.array(X.T)
        self.X = StandardScaler().fit_transform(X)
        orderings = np.array(orderings) - 1
        self.Y = ranking_ordering_conversion(orderings)
        self.__check_dataset_validity__()

    def get_single_train_test_split(self):
        cv_iter = ShuffleSplit(n_splits=1, test_size=0.3, random_state=self.random_state)
        (train_idx, test_idx) = list(cv_iter.split(self.X))[0]
        return self.X[train_idx], self.Y[train_idx], self.X[test_idx], self.Y[test_idx]

    def get_dataset_dictionaries(self):
        return self.X, self.Y
