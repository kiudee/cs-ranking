import logging
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
        self.logger = logging.getLogger(SushiDatasetReader.__name__)

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
        feature_objects = np.array(feature_objects, dtype=float)
        feature_users = np.array(feature_users, dtype=float)
        for i in range(len(feature_objects)):
            feature_objects[i][0:3] = np.log(feature_objects[i][0:3] + 2)
        feature_users = np.log(feature_users + 2)
        X = []
        rankings = []
        Xc = []
        i = 0
        for ranking1, ranking2 in zip(rankings_a, rankings_b):
            user_f = np.array([feature_users[i] for _ in range(len(ranking1))])

            objects = feature_objects[ranking1, :]
            ranking = np.arange(len(ranking1))
            np.random.shuffle(ranking)
            objects = objects[ranking, :]
            # objects = np.hstack((user_f, objects))
            X.append(objects)
            rankings.append(ranking)

            objects = feature_objects[ranking2, :]
            ranking = np.arange(len(ranking2))
            np.random.shuffle(ranking)
            objects = objects[ranking, :]
            # objects = np.hstack((user_f, objects))
            X.append(objects)
            rankings.append(ranking)

            Xc.append(feature_users[i])
            i += 1

        self.X = np.array(X)
        self.Y = np.array(rankings)
        if self.learning_problem == DYAD_RANKING:
            self.Xc = np.array(Xc)
            self.logger(Xc.shape)
        else:
            self.Xc = None
