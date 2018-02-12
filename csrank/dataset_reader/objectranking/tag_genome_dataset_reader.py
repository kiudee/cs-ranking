import logging
import os
from itertools import combinations, product

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state

from csrank.constants import OBJECT_RANKING
from csrank.util import scores_to_rankings
from ..dataset_reader import DatasetReader
from ..objectranking.util import initialize_similarity_matrix, get_key_for_indices, weighted_cosine_similarity

TAG_POPULARITY = "TagPopularity"
DOC_FREQUENCY = "DocFrequency"


class TagGenomeDatasetReader(DatasetReader):
    def __init__(self, n_train_instances=10000, n_test_instances=10000, n_objects=5, random_state=None, **kwargs):
        super(TagGenomeDatasetReader, self).__init__(learning_problem=OBJECT_RANKING, dataset_folder='tag_genome',
                                                     **kwargs)

        movies_dat = os.path.join(self.dirname, 'movies.dat')
        tag_rel_dat = os.path.join(self.dirname, 'tag_relevance.dat')
        tag_dat = os.path.join(self.dirname, 'tags.dat')
        movies_df = pd.read_csv(movies_dat, "\t")
        tag_rel_df = pd.read_csv(tag_rel_dat, "\t")
        tag_df = pd.read_csv(tag_dat, "\t")

        self.type = OBJECT_RANKING
        self.Xc = None
        self.movies_file = os.path.join(self.dirname, 'movies_transformed.csv')
        self.similarity_matrix_file = os.path.join(self.dirname, 'similarity_matrix.csv')
        self.tag_file = os.path.join(self.dirname, 'tags_info.csv')

        self.n_objects = n_objects
        self.n_features = 1128
        self.n_test_instances = n_test_instances
        self.n_train_instances = n_train_instances
        self.logger = logging.getLogger(name='TagGenomeDataset')
        self.random_state = check_random_state(random_state)

        if (not (os.path.isfile(self.movies_file) and os.path.isfile(self.similarity_matrix_file) and os.path.isfile(
                self.tag_file))):
            self.__load_dataset__(movies_df, tag_rel_df, tag_df)

        self.similarity_matrix_lin_list = initialize_similarity_matrix(self.similarity_matrix_file)
        self.movies_df = pd.read_csv(self.movies_file)
        self.logger.debug("Done creating the complete dataset")

    def __load_dataset__(self, movies_df, tag_rel_df, tag_df):
        objects = []
        movies = movies_df.as_matrix()
        tags_rel = tag_rel_df.as_matrix()
        for i, movie in enumerate(movies):
            a = i * self.n_features
            b = (i + 1) * self.n_features
            objects.append(tags_rel[a:b, 2])
        self.logger.debug("Done loading the features for the movies")
        objects = np.array(objects)
        doc_freqs = []
        for i, tag in enumerate(tag_df.values):
            df = tag_rel_df.loc[tag_rel_df['TagID'] == tag[0]]
            movies_df[tag[1]] = objects[:, i]
            freq = np.sum(np.array(df["Relevance"]) >= 0.5) + 2
            doc_freqs.append(freq)
        self.logger.debug("Done loading the DocFrequency for the tags")
        tag_df[DOC_FREQUENCY] = doc_freqs

        movies_df.to_csv(self.movies_file)
        tag_df.to_csv(self.tag_file)

        num_of_movies = movies.shape[0]
        combinations_list = np.array(list(combinations(range(num_of_movies), 2)))
        similarity_matrix_lin_list = dict()

        movie_features = movies_df.as_matrix()[:, 3:]
        weights = np.log(np.array(tag_df[TAG_POPULARITY])) / np.log(np.array(tag_df[DOC_FREQUENCY]))

        for i, j in combinations_list:
            similarity_matrix_lin_list[get_key_for_indices(i, j)] = weighted_cosine_similarity(weights,
                                                                                               movie_features[i],
                                                                                               movie_features[j])
            # self.logger.info("calculating similarity {},{},{}".format(i, j, sim))
        for i in range(num_of_movies):
            similarity_matrix_lin_list[get_key_for_indices(i, i)] = 1.0
        series = pd.Series(similarity_matrix_lin_list)
        matrix_df = pd.DataFrame({'col_major_index': series.index, 'similarity': series.values})
        matrix_df.to_csv(self.similarity_matrix_file)
        self.logger.debug("Done calculating the similarity matrix stored at: {}".format(self.similarity_matrix_file))

    def make_similarity_based_dataset(self, n_instances, seed=42):

        random_state = np.random.RandomState(seed=seed)
        X = np.empty((n_instances, self.n_objects, self.n_features), dtype=float)
        rankings = np.empty((n_instances, self.n_objects), dtype=int)
        similarity_scores = np.empty_like(rankings, dtype=float)
        movie_features = self.movies_df.as_matrix()[:, 4:]
        for i in range(n_instances):
            subset = random_state.choice(movie_features.shape[0], size=self.n_objects, replace=False)
            X[i] = movie_features[subset]
            query = random_state.choice(self.n_objects, size=1)
            one_row = [self.similarity_matrix_lin_list[get_key_for_indices(i, j)] for i, j in
                       product(subset[query], subset)]
            similarity_scores[i] = np.array(one_row)

        # Higher the similarity lower the rank of the object
        rankings = scores_to_rankings(similarity_scores)

        for i, x in enumerate(X):
            x = StandardScaler().fit_transform(x)
            X[i] = x
        return X, rankings

    def splitter(self, iter):
        for i in iter:
            X_train, Y_train = self.make_similarity_based_dataset(self.n_train_instances, seed=10 * i + 32)
            X_test, Y_test = self.make_similarity_based_dataset(self.n_test_instances, seed=10 * i + 32)
        yield X_train, Y_train, X_test, Y_test

    def get_complete_dataset(self):
        pass

    def get_train_test_datasets(self, n_datasets=5):
        splits = np.array(n_datasets)
        return self.splitter(splits)

    def get_single_train_test_split(self):
        seed = self.random_state.randint(2 ** 32, dtype='uint32')
        self.X, self.rankings = X_train, Y_train = self.make_similarity_based_dataset(self.n_train_instances, seed=seed)
        self.__check_dataset_validity__()

        self.X, self.rankings = X_test, Y_test = self.make_similarity_based_dataset(self.n_test_instances,
                                                                                    seed=seed + 1)
        self.__check_dataset_validity__()
        return X_train, Y_train, X_test, Y_test
