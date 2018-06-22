import collections
import glob
import logging
import os
from itertools import combinations, product

import h5py
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state

from csrank.constants import OBJECT_RANKING
from csrank.numpy_util import scores_to_rankings
from ..dataset_reader import DatasetReader
from ..util import get_similarity_matrix, get_key_for_indices, distance_metric_multilabel


class ImageDatasetReader(DatasetReader):
    def __init__(self, n_train_instances=10000, n_test_instances=10000, n_objects=5, random_state=None, **kwargs):
        super(ImageDatasetReader, self).__init__(learning_problem=OBJECT_RANKING, dataset_folder='image_dataset',
                                                 **kwargs)
        TRAIN_LABEL = 'gt_label_train'
        VAL_LABEL = 'gt_label_val'
        # TEST_LABEL = 'gt_label_test'

        # self.test_file = os.path.join(dirname, DATASET_FOLDER, "test.hd5")
        # self.labels_test_files = glob.glob(os.path.join(dirname, DATASET_FOLDER, TEST_LABEL, "*.txt"))
        # self.labels_test_files.sort()
        self.logger = logging.getLogger(ImageDatasetReader.__name__)

        self.train_file = os.path.join(self.dirname, "train.hd5")
        self.similarity_matrix_train_file = os.path.join(self.dirname, "train_similarity_matrix.csv")
        self.labels_train_files = glob.glob(os.path.join(self.dirname, TRAIN_LABEL, "*.txt"))
        self.labels_train_files.sort()

        self.test_file = os.path.join(self.dirname, "val.hd5")
        self.similarity_matrix_test_file = os.path.join(self.dirname, "val_similarity_matrix.csv")
        self.labels_test_files = glob.glob(os.path.join(self.dirname, VAL_LABEL, "*.txt"))
        self.labels_test_files.sort()

        self.n_test_instances = n_test_instances
        self.n_train_instances = n_train_instances
        self.n_objects = n_objects
        self.n_features = 2048
        self.label_names = []
        for file in self.labels_test_files:
            self.label_names.append(os.path.basename(file).split('.')[0].split('_')[0])

        self.logger.info('Image Features Train File {}'.format(self.train_file))
        self.logger.info('Image Features Test File {}'.format(self.test_file))
        self.logger.info('Image Labels Train File {}'.format(self.labels_train_files))
        self.logger.info('Image Labels Test File {}'.format(self.labels_test_files))
        self.logger.info('Image Labels Names {}'.format(self.label_names))
        self.random_state = check_random_state(random_state)
        if (not (os.path.isfile(self.similarity_matrix_test_file) and
                 os.path.isfile(self.similarity_matrix_train_file))):
            self.__load_dataset__()

    def __load_dataset__(self):
        self.create_multilabel_dataset(datatype='train')
        self.create_multilabel_dataset(datatype='test')

    def splitter(self, iter):
        for i in iter:
            self.X, self.Y = X_train, Y_train = self.make_similarity_based_dataset(datatype='train', seed=10 * i + 32)
            self.__check_dataset_validity__()

            self.X, self.Y = X_test, Y_test = self.make_similarity_based_dataset(datatype='test', seed=10 * i + 32)
            self.__check_dataset_validity__()

        yield X_train, Y_train, X_test, Y_test

    def get_dataset_dictionaries(self):
        pass

    def get_train_test_datasets(self, n_datasets=5):
        splits = np.array(n_datasets)
        return self.splitter(splits)

    def get_single_train_test_split(self):
        seed = self.random_state.randint(2 ** 32, dtype='uint32')
        self.X, self.Y = X_train, Y_train = self.make_similarity_based_dataset(datatype='train', seed=seed)
        self.__check_dataset_validity__()

        self.X, self.Y = X_test, Y_test = self.make_similarity_based_dataset(datatype='test', seed=seed + 1)
        self.__check_dataset_validity__()
        return X_train, Y_train, X_test, Y_test

    def make_similarity_based_dataset(self, datatype='train', seed=42):
        """Picks a random subset of objects, determines the medoid and ranks the objects
        based on the distance to the medoid.

        The medoid is also included in the ordering."""
        random_state = np.random.RandomState(seed=seed)
        if datatype == 'train':
            image_features = self.image_features_train
            n_instances = self.n_train_instances
            similarity_matrix_file = self.similarity_matrix_train_file
        elif datatype == 'test':
            image_features = self.image_features_test
            n_instances = self.n_test_instances
            similarity_matrix_file = self.similarity_matrix_test_file

        X = np.empty((n_instances, self.n_objects, self.n_features), dtype=float)
        similarity_scores = np.empty((n_instances, self.n_objects), dtype=float)
        similarity_matrix_lin_list = get_similarity_matrix(similarity_matrix_file)

        for i in range(n_instances):
            subset = random_state.choice(image_features.shape[0], size=self.n_objects, replace=False)
            X[i] = image_features[subset]
            query = random_state.choice(self.n_objects, size=1)
            one_row = [similarity_matrix_lin_list[get_key_for_indices(i, j)] for i, j in product(subset[query], subset)]
            similarity_scores[i] = np.array(one_row)

        Y = scores_to_rankings(similarity_scores)
        for i, x in enumerate(X):
            x = StandardScaler().fit_transform(x)
            X[i] = x
        return X, Y

    def create_multilabel_dataset(self, datatype='train'):
        if datatype == 'train':
            image_file_name = self.train_file
            label_file_names = self.labels_train_files
            similarity_matrix_file = self.similarity_matrix_train_file
        elif datatype == 'test':
            image_file_name = self.test_file
            label_file_names = self.labels_test_files
            similarity_matrix_file = self.similarity_matrix_test_file

        image_file = h5py.File(image_file_name, "r")
        names = np.array(image_file.get('name')[()], dtype=str)
        image_features = np.array(image_file.get('feature'))

        names = [str.replace(w, 'b', '') for w in names]
        names = [str.replace(w, '\n', '') for w in names]
        label_vectors_dictionary = {key: [] for key in names}

        label_keys = []
        for file in label_file_names:
            lines = np.array([line.rstrip('\n') for line in open(file)])
            for line in lines:
                label_keys.append(line.split(' ')[0])
                if (line.split(' ')[0] in label_vectors_dictionary.keys()):
                    label_vectors_dictionary[line.split(' ')[0]].append(int(line.split(' ')[-1]))
        label_vectors_dictionary = collections.OrderedDict(sorted(label_vectors_dictionary.items()))
        label_vectors = []
        for key, value in label_vectors_dictionary.items():
            label_vectors.append(value)
        label_keys = list(set(label_keys))
        label_keys.sort()
        label_vectors = np.array(label_vectors)
        image_features = StandardScaler().fit_transform(image_features)

        num_of_images = label_vectors.shape[0]
        combinations_list = np.array(list(combinations(range(num_of_images), 2)))
        similarity_matrix_lin_list = dict()

        for i, j in combinations_list:
            similarity_matrix_lin_list[get_key_for_indices(i, j)] = distance_metric_multilabel(
                label_vectors[i], label_vectors[j], image_features[i], image_features[j])
        # self.logger.info("calculating similarity {},{},{}".format(i, j, sim))

        for i in range(num_of_images):
            similarity_matrix_lin_list[get_key_for_indices(i, i)] = 1.0

        series = pd.Series(similarity_matrix_lin_list)
        matrix_df = pd.DataFrame({'col_major_index': series.index, 'similarity': series.values})
        matrix_df.to_csv(similarity_matrix_file)
        self.logger.debug("Done calculating the similarity matrix stored at: {}".format(similarity_matrix_file))
        assert self.n_features == image_features.shape[1], "Number of features not correct in dataset {}".format(
            image_features.shape[1])
        return image_features, label_vectors
