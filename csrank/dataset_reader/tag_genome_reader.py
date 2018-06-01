import logging
import os
from abc import ABCMeta, abstractmethod
from itertools import combinations, product

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state

from csrank.dataset_reader.dataset_reader import DatasetReader
from .util import get_key_for_indices, get_similarity_matrix, print_no_newline, weighted_cosine_similarity, \
    critique_dist

MOVIE_ID = 'movieId'

TAG_POPULARITY = 'tagpopularity'
DOC_FREQUENCY = 'docfrequency'


class TagGenomeDatasetReader(DatasetReader, metaclass=ABCMeta):
    def __init__(self, n_train_instances=10000, n_test_instances=10000, n_objects=5, random_state=None, **kwargs):
        super(TagGenomeDatasetReader, self).__init__(dataset_folder='movie_lens', **kwargs)

        self.logger = logging.getLogger(TagGenomeDatasetReader.__name__)
        genome_scores = os.path.join(self.dirname, 'genome-scores.csv')
        genome_tags = os.path.join(self.dirname, 'genome-tags.csv')
        movies = os.path.join(self.dirname, 'movies.csv')
        tag_rel_df = pd.read_csv(genome_scores)
        tag_info_df = pd.read_csv(genome_tags)
        movies_df = pd.read_csv(movies)

        self.movies_file = movies
        self.similarity_matrix_file = os.path.join(self.dirname, 'similarity_matrix.csv')
        self.n_objects = n_objects
        self.n_features = np.array(tag_info_df['tagId']).shape[0]
        self.n_test_instances = n_test_instances
        self.n_train_instances = n_train_instances
        self.random_state = check_random_state(random_state)
        self.model = None
        self.weights = np.log(np.array(tag_info_df[TAG_POPULARITY])) / np.log(np.array(tag_info_df[DOC_FREQUENCY]))
        self.knn_data_file = os.path.join(self.dirname, 'nearest_neighbors_data.npy')
        if not (os.path.isfile(self.similarity_matrix_file) and os.path.isfile(self.knn_data_file)):
            self.__load_dataset__(tag_rel_df, tag_info_df, movies_df)

        self.similarity_matrix = get_similarity_matrix(self.similarity_matrix_file)
        self.movies_df = pd.read_csv(self.movies_file)
        self.n_movies = len(self.movies_df)
        self.logger.debug("Done creating the complete dataset")

    def __load_dataset__(self, tag_rel_df, tag_info_df, movies_df):
        objects = []
        tags_rel = tag_rel_df.as_matrix()
        movie_ids = np.unique(np.array(tag_rel_df[MOVIE_ID]))
        for i, movie_id in enumerate(movie_ids):
            a = i * self.n_features
            b = (i + 1) * self.n_features
            objects.append(tags_rel[a:b, 2])
        objects = np.array(objects)
        movies_df = movies_df[movies_df[MOVIE_ID].isin(movie_ids)]

        for i, tag in enumerate(tag_info_df.values):
            movies_df[tag[1]] = objects[:, i]
        movies_df.to_csv(self.movies_file, index=False)

        self.logger.debug("Done loading the features for the movies")

        num_of_movies = movie_ids.shape[0]
        combinations_list = np.array(list(combinations(range(num_of_movies), 2)))
        similarity_matrix = dict()

        features = movies_df.as_matrix()[:, 3:]

        for i, j in combinations_list:
            similarity_matrix[get_key_for_indices(i, j)] = weighted_cosine_similarity(self.weights)(features[i],
                                                                                                    features[j])
            self.logger.info(
                "Calculating similarity {},{}, {}".format(i, j, similarity_matrix[get_key_for_indices(i, j)]))

        for i in range(num_of_movies):
            similarity_matrix[get_key_for_indices(i, i)] = 1.0

        series = pd.Series(similarity_matrix)
        matrix_df = pd.DataFrame({'col_major_index': series.index, 'similarity': series.values})
        matrix_df.to_csv(self.similarity_matrix_file, index=False)
        self.logger.debug("Done calculating the similarity matrix stored at: {}".format(self.similarity_matrix_file))

        neighbours = []
        for i, f in enumerate(features):
            scores = np.array([similarity_matrix[get_key_for_indices(i, j)] for j in range(num_of_movies)])
            neighbours.append(np.array([scores, np.argsort(scores)[::-1]]))
            print_no_newline(i, len(features))
        neighbours = np.array(neighbours)
        np.save(self.knn_data_file, neighbours)
        self.logger.debug("Done calculating the nearest neighbours data stored at: {}".format(self.knn_data_file))

    @abstractmethod
    def make_similarity_based_dataset(self, n_instances, n_objects, seed, **kwargs):
        self.logger.info('For instances {} objects {}, seed {}'.format(n_instances, n_objects, seed))
        random_state = check_random_state(seed)
        X = np.empty((n_instances, n_objects, self.n_features), dtype=float)
        scores = np.empty((n_instances, n_objects), dtype=float)
        movie_features = self.movies_df.as_matrix()[:, 3:]
        for i in range(n_instances):
            subset = random_state.choice(movie_features.shape[0], size=n_objects, replace=False)
            X[i] = movie_features[subset]
            query = subset[0]
            while query in subset:
                query = random_state.choice(movie_features.shape[0], size=1)
            one_row = [self.similarity_matrix[get_key_for_indices(i, j)] for i, j in product(query, subset)]
            scores[i] = np.array(one_row)
        return X, scores

    @abstractmethod
    def make_nearest_neighbour_dataset(self, n_instances, n_objects, seed, **kwargs):
        self.logger.info('For instances {} objects {}, seed {}'.format(n_instances, n_objects, seed))
        random_state = check_random_state(seed)
        length = (int(n_instances / self.n_movies) + 1)
        n_neighbors = length * n_objects
        movie_features = self.movies_df.as_matrix()[:, 3:]
        nearest_neighbors_data = np.load(self.knn_data_file)

        similarity_scores = nearest_neighbors_data[:, 0, 1:n_neighbors + 1]
        set_of_movies = nearest_neighbors_data[:, 1, 1:n_neighbors + 1].astype(int)
        set_of_movies = set_of_movies.reshape(self.n_movies * length, n_objects)
        similarity_scores = similarity_scores.reshape(self.n_movies * length, n_objects)

        indices = np.arange(self.n_movies * length)
        random_state.shuffle(indices)
        set_of_movies = set_of_movies[indices]
        similarity_scores = similarity_scores[indices]

        X = np.zeros((n_instances, n_objects, self.n_features), dtype=float)
        scores = np.empty((n_instances, n_objects), dtype=float)
        for i in range(n_instances):
            indices = np.arange(n_objects)
            random_state.shuffle(indices)
            X[i] = movie_features[set_of_movies[i]][indices]
            scores[i] = similarity_scores[i][indices]

        return X, scores

    @abstractmethod
    def make_critique_fit_dataset(self, n_instances, n_objects, seed, direction, **kwargs):
        self.logger.info('For instances {} objects {}, seed {}, direction {}'.format(n_instances, n_objects, seed, direction))
        random_state = check_random_state(seed)
        length = (int(n_instances / self.n_movies) + 1)
        movie_features = self.movies_df.as_matrix()[:, 3:]
        X = []
        scores = []
        for i, feature in enumerate(movie_features):
            tag_ids = np.where(np.logical_and((feature > 0.5), (feature < 0.6)))[0]
            tag_ids = tag_ids[0:length]
            distances = [self.similarity_matrix[get_key_for_indices(i, j)] for j in range(self.n_movies)]
            distances = np.array(distances)
            critique_d = critique_dist(feature, movie_features, tag_ids, direction=direction)
            critique_fit = np.multiply(critique_d, distances)
            orderings = np.argsort(critique_fit, axis=1)[:, ::-1][:, 0:n_objects]
            for j, ordering in enumerate(orderings):
                X.append(movie_features[ordering])
                scores.append(critique_fit[j, ordering])
        X = np.array(X)
        scores = np.array(scores)
        indices = random_state.choice(X.shape[0], n_instances, replace=False)
        X = X[indices]
        scores = scores[indices]
        return X, scores

    def get_dataset_dictionaries(self, lengths=[5, 6]):
        X_train = dict()
        Y_train = dict()
        X_test = dict()
        Y_test = dict()
        for n_obj in lengths:
            seed = self.random_state.randint(2 ** 32, dtype='uint32')
            total_instances = self.n_test_instances + self.n_train_instances
            X, Y = self.dataset_function(total_instances, n_obj, seed=seed)

        return X_train, Y_train, X_test, Y_test

    def get_train_test_datasets(self, n_datasets=5):
        splits = np.array(n_datasets)
        return self.splitter(splits)

    def get_single_train_test_split(self):
        seed = self.random_state.randint(2 ** 32, dtype='uint32')
        total_instances = self.n_test_instances + self.n_train_instances
        self.X, self.Y = self.dataset_function(total_instances, self.n_objects, seed=seed)
        self.__check_dataset_validity__()
        return train_test_split(self.X, self.Y, random_state=self.random_state, test_size=self.n_test_instances)

    def splitter(self, iter):
        for i in iter:
            seed = self.random_state.randint(2 ** 32, dtype='uint32') + i
            total_instances = self.n_test_instances + self.n_train_instances
            self.X, self.Y = self.dataset_function(total_instances, self.n_objects, seed=seed)
            self.__check_dataset_validity__()
        yield train_test_split(self.X, self.Y, random_state=self.random_state, test_size=self.n_test_instances)
