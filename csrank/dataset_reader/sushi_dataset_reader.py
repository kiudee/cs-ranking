import os
from abc import ABCMeta

import numpy as np
import pandas as pd

from csrank.constants import DYAD_RANKING
from .dataset_reader import DatasetReader


class SushiDatasetReader(DatasetReader, metaclass=ABCMeta):
    def __init__(self, **kwargs):
        super(SushiDatasetReader, self).__init__(dataset_folder='sushi', **kwargs)
        self.object_datafile = os.path.join(self.dirname, 'sushi3.txt')
        self.user_datafile = os.path.join(self.dirname, 'sushi3_u.txt')
        self.rankings_small = os.path.join(self.dirname, 'sushi_small.txt')
        self.rankings_big = os.path.join(self.dirname, 'sushi_big.txt')
        self.__load_dataset__()
        self.__check_dataset_validity__()

    def __load_dataset__(self):
        # Code to concatenate the context feature to each object feature
        # c = np.tile(f1[np.newaxis, :], (f2.shape[0], 1))
        # combined = np.concatenate((c, f2), axis=1)
        objects = pd.read_csv(self.object_datafile, sep='\t')
        users = pd.read_csv(self.user_datafile, sep='\t')

        small_rankings = pd.read_csv(self.rankings_small, sep=' ')
        big_rankings = pd.read_csv(self.rankings_big, sep=' ')
        rankings_a = small_rankings.as_matrix()[:, 2:13]
        rankings_b = big_rankings.as_matrix()[:, 2:13]

        feature_objects = objects.as_matrix()[:, 2:10]
        feature_users = users.as_matrix()[:, 1:12]
        feature_objects[:, 2] = [int("{0:b}".format(i)) for i in feature_objects[:, 2]]

        X = []
        rankings = []
        Xc = []
        for i, ranking in enumerate(rankings_a):
            combined = feature_objects[ranking, :]
            ranking = np.arange(len(ranking))
            np.random.shuffle(ranking)
            combined = combined[ranking, :]
            X.append(combined)
            rankings.append(ranking)
            Xc.append(feature_users[i])

        for i, ranking in enumerate(rankings_b):
            combined = feature_objects[ranking, :]
            ranking = np.arange(len(ranking))
            np.random.shuffle(ranking)
            combined = combined[ranking, :]
            X.append(combined)
            rankings.append(ranking)
            Xc.append(feature_users[i])

        self.X = np.array(X)
        self.Y = np.array(rankings)
        if (self.learning_problem == DYAD_RANKING):
            self.Xc = np.array(Xc)
        else:
            self.Xc = None
