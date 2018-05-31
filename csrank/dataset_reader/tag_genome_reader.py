import logging
import os
from abc import ABCMeta, abstractmethod
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state

from csrank.dataset_reader.dataset_reader import DatasetReader
from .util import get_key_for_indices, weighted_cosine_similarity

MOVIE_ID = 'movieId'

TAG_POPULARITY = 'tagpopularity'
DOC_FREQUENCY = 'docfrequency'


class TagGenomeDatasetReader(DatasetReader, metaclass=ABCMeta):
    def __init__(self, dataset_type="similarity", n_train_instances=10000, n_test_instances=10000, n_objects=5,
                 random_state=None, **kwargs):
        super(TagGenomeDatasetReader, self).__init__(dataset_folder='movie_lens', **kwargs)
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
        self.logger = logging.getLogger(TagGenomeDatasetReader.__name__)
        self.random_state = check_random_state(random_state)
        self.model = None
        dataset_func_dict = {"similarity": self.make_similarity_based_dataset,
                             "nearest_neighbour": self.make_nearest_neighbour_dataset}
        if dataset_type not in dataset_func_dict.keys():
            dataset_type = "similarity"

        self.dataset_function = dataset_func_dict[dataset_type]
        self.weights = np.log(np.array(tag_info_df[TAG_POPULARITY])) / np.log(np.array(tag_info_df[DOC_FREQUENCY]))
        if not os.path.isfile(self.similarity_matrix_file):
            self.__load_dataset__(tag_rel_df, tag_info_df, movies_df)

        # self.similarity_matrix_lin_list = get_similarity_matrix(self.similarity_matrix_file)
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
            sim = similarity_matrix[get_key_for_indices(i, j)] = weighted_cosine_similarity(self.weights)(features[i],
                                                                                                          features[j])
            self.logger.info("calculating similarity {},{},{}".format(i, j, sim))
        for i in range(num_of_movies):
            similarity_matrix[get_key_for_indices(i, i)] = 1.0

        series = pd.Series(similarity_matrix)
        matrix_df = pd.DataFrame({'col_major_index': series.index, 'similarity': series.values})
        matrix_df.to_csv(self.similarity_matrix_file, index=False)
        self.logger.debug("Done calculating the similarity matrix stored at: {}".format(self.similarity_matrix_file))

    @abstractmethod
    def make_similarity_based_dataset(self, n_instances, n_objects, seed):
        pass

    @abstractmethod
    def make_nearest_neighbour_dataset(self, n_instances, n_objects, seed):
        pass

    @abstractmethod
    def make_critique_fit_dataset(self, n_instances, n_objects, seed):
        pass

    def get_dataset_dictionaries(self, lengths=[5, 6]):
        pass

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
