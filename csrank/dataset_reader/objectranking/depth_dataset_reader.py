import os
import re
from collections import namedtuple

import numpy as np
from scipy.stats import rankdata
from sklearn.utils import check_random_state

from csrank.constants import OBJECT_RANKING
from .util import sub_sampling_rankings
from ..dataset_reader import DatasetReader

__all__ = ['DepthDatasetReader']


class DepthDatasetReader(DatasetReader):
    def __init__(self, dataset_type='deep', random_state=None, **kwargs):
        super(DepthDatasetReader, self).__init__(learning_problem=OBJECT_RANKING, dataset_folder='depth_data', **kwargs)
        options = {'deep': ['complete_deep_train.dat', 'complete_deep_test.dat'],
                   'basic': ['saxena_basic61x55.dat', 'saxena_basicTest61x55.dat'],
                   'semantic': ['saxena_semantic61x55.dat', 'saxena_semanticTest61x55.dat']
                   }
        if dataset_type not in options:
            dataset_type = 'deep'
        train_filename, test_file_name = options[dataset_type]
        self.train_file = os.path.join(self.dirname, train_filename)
        self.test_file = os.path.join(self.dirname, test_file_name)
        self.random_state = check_random_state(random_state)

        self.__load_dataset__()

    def __load_dataset__(self):
        self.x_train, self.depth_train = load_dataset(self.train_file)
        self.x_test, self.depth_test = load_dataset(self.test_file)

    def get_train_test_datasets(self, n_datasets=5):
        splits = np.array(n_datasets)
        return self.splitter(splits)

    def get_single_train_test_split(self):
        seed = self.random_state.randint(2 ** 32, dtype='uint32')
        X_train, Y_train = self.get_train_dataset_sampled_partial_rankings(seed=seed)
        X_train, Y_train = sub_sampling_rankings(X_train, Y_train, n_objects=5)
        X_test, Y_test = self.get_test_dataset_ties()
        return X_train, Y_train, X_test, Y_test

    def get_dataset_dictionaries(self):
        pass

    def splitter(self, iter):
        for i in iter:
            X_train, Y_train = self.get_train_dataset_sampled_partial_rankings(seed=10 * i + 32)
            X_test, Y_test = self.get_test_dataset_ties()
        yield X_train, Y_train, X_test, Y_test

    def get_test_dataset_sampled_partial_rankings(self, **kwargs):
        self.X, self.Y = self.get_dataset_sampled_partial_rankings(datatype='test', **kwargs)
        self.__check_dataset_validity__()
        return self.X, self.Y

    def get_train_dataset_sampled_partial_rankings(self, **kwargs):
        self.X, self.Y = self.get_dataset_sampled_partial_rankings(datatype='train', **kwargs)
        self.__check_dataset_validity__()
        return self.X, self.Y

    def get_test_dataset(self):
        self.X, self.Y = self.get_dataset(datatype='test')
        self.__check_dataset_validity__()
        return self.X, self.Y

    def get_train_dataset(self):
        self.X, self.Y = self.get_dataset(datatype='train')
        self.__check_dataset_validity__()
        return self.X, self.Y

    def get_test_dataset_ties(self):
        self.X, self.Y = self.get_dataset_ties(datatype='test')
        self.__check_dataset_validity__()
        return self.X, self.Y

    def get_train_dataset_ties(self):
        self.X, self.Y = self.get_dataset_ties(datatype='train')
        self.__check_dataset_validity__()
        return self.X, self.Y

    def get_dataset_sampled_partial_rankings(self, datatype='train', max_number_of_rankings_per_image=10, seed=42):
        random_state = np.random.RandomState(seed=seed)
        x_train, depth_train = self.get_deep_copy_dataset(datatype)
        X = []
        rankings = []
        order_lengths = np.array(
            [len(np.unique(depths[np.where(depths <= 0.80)[0]], return_index=True)[1]) for depths in depth_train])
        order_length = np.min(order_lengths)

        for features, depths in zip(x_train, depth_train):
            value, obj_indices = np.unique(depths[np.where(depths <= 0.80)[0]], return_index=True)
            interval = int(len(obj_indices) / order_length)
            if interval < max_number_of_rankings_per_image:
                num_of_orderings_per_image = interval
            else:
                num_of_orderings_per_image = max_number_of_rankings_per_image
            objects_i = np.empty([order_length, num_of_orderings_per_image], dtype=int)
            for i in range(order_length):
                if i != order_length - 1:
                    objs = random_state.choice(obj_indices[i * interval:(i + 1) * interval], num_of_orderings_per_image,
                                               replace=False)
                else:
                    objs = random_state.choice(obj_indices[i * interval:len(obj_indices)], num_of_orderings_per_image,
                                               replace=False)
                objects_i[i] = objs
            for i in range(num_of_orderings_per_image):
                indices = objects_i[:, i]
                np.random.shuffle(indices)
                X.append(features[indices])
                ranking = rankdata(depths[indices]) - 1
                rankings.append(ranking)
        X = np.array(X)
        rankings = np.array(rankings)
        return X, rankings

    def get_dataset(self, datatype='train'):
        x, y = self.get_deep_copy_dataset(datatype)
        X = []
        rankings = []
        for features, depths in zip(x, y):
            value, indices = np.unique(depths, return_index=True)
            np.random.shuffle(indices)
            X.append(features[indices])
            ranking = rankdata(depths[indices]) - 1
            rankings.append(ranking)
        X = np.array(X)
        rankings = np.array(rankings)
        return X, rankings

    def get_dataset_ties(self, datatype='train'):
        X, y = self.get_deep_copy_dataset(datatype)
        for depth in y:
            depth[np.where(depth >= 0.80)[0]] = 0.80
        rankings = np.array([rankdata(depth) - 1 for depth in y])
        return X, rankings

    def get_deep_copy_dataset(self, datatype):
        if datatype == 'train':
            x, y = np.copy(self.x_train), np.copy(self.depth_train)
        elif datatype == 'test':
            x, y = np.copy(self.x_test), np.copy(self.depth_test)
        return x, y


def load_dataset(filename):
    Instance = namedtuple('Instance', ['depth', 'features'])
    instances = dict()
    with open(filename) as f:
        for line in f:
            arr = line.split()
            depth = float(arr[0])
            qid = int(arr[1][4:])
            if qid not in instances:
                instances[qid] = []
                if '#' in arr:
                    arr = arr[:-2]
            features = [float(re.search('\:([0-9\.]*)', x).group(1)) for x in arr[2:]]
            instances[qid].append(Instance(depth, features))
    n_instances = len(instances)
    n_objects = len(instances[1])
    n_features = len(instances[1][0][1])
    X = np.empty((n_instances, n_objects, n_features))
    y = np.empty((n_instances, n_objects))
    for i, inst in enumerate(instances.values()):
        for j, (depth, features) in enumerate(inst):
            X[i, j] = features
            y[i, j] = depth
        ind = np.argsort(y[i, :])
        y[i] = y[i, ind]
        X[i] = X[i, ind]
    return X, y
