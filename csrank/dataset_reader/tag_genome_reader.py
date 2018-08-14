import logging
import os
from abc import ABCMeta, abstractmethod
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state

from csrank.dataset_reader.dataset_reader import DatasetReader
from .util import get_key_for_indices, get_similarity_matrix, standardize_features

MOVIE_ID = 'movieId'
TAG_ID = 'tagId'
TAG = 'tag'
TAG_POPULARITY = 'tagpopularity'
DOC_FREQUENCY = 'docfrequency'
RELEVANCE = "relevance"


class TagGenomeDatasetReader(DatasetReader, metaclass=ABCMeta):
    def __init__(self, n_train_instances=10000, n_test_instances=10000, n_objects=5, random_state=None,
                 standardize=True, **kwargs):
        super(TagGenomeDatasetReader, self).__init__(dataset_folder='movie_lens', **kwargs)

        self.logger = logging.getLogger(TagGenomeDatasetReader.__name__)

        self.movies_file = os.path.join(self.dirname, 'movies.csv')
        genome_scores = pd.read_csv(os.path.join(self.dirname, 'genome-scores.csv'))
        tags_applies = pd.read_csv(os.path.join(self.dirname, 'tags.csv'))

        self.tags_info_file = os.path.join(self.dirname, 'genome-tags.csv')
        genome_tags = pd.read_csv(self.tags_info_file)
        movies_df = pd.read_csv(self.movies_file)
        self.similarity_matrix_file = os.path.join(self.dirname, 'similarity_matrix.csv')
        self.standardize = standardize
        self.n_objects = n_objects
        self.n_features = np.array(genome_tags[TAG_ID]).shape[0]
        self.n_test_instances = n_test_instances
        self.n_train_instances = n_train_instances
        self.random_state = check_random_state(random_state)
        if not os.path.isfile(self.similarity_matrix_file):
            self.__load_dataset__(genome_scores, genome_tags, tags_applies, movies_df)

        self.logger.info("Loading similarity matrix")
        self.similarity_matrix = get_similarity_matrix(self.similarity_matrix_file)
        self.logger.info("Done loading similarity matrix")
        self.weights = np.log(np.array(genome_tags[TAG_POPULARITY])) / np.log(np.array(genome_tags[DOC_FREQUENCY]))
        self.movies_df = pd.read_csv(self.movies_file)
        self.n_movies = len(self.movies_df)
        self.movie_features = self.movies_df.as_matrix()[:, 3:].astype(float)
        self.tags_info_df = pd.read_csv(self.tags_info_file)
        self.logger.info("Done creating the complete dataset")

    def __load_dataset__(self, genome_scores, genome_tags, tags_applies, movies_df):
        tags_applies[TAG] = tags_applies[TAG].str.lower()
        values_with_pop = []
        for tid, t in genome_tags.as_matrix():
            tags_a = tags_applies.loc[tags_applies[TAG] == t.lower()]
            tags_total = np.array(tags_a[MOVIE_ID])
            values_with_pop.append([tid, t, len(tags_total)])
            self.logger.info('Tag popularity for tag {} is: {}'.format(t, len(tags_total)))
            if len(tags_total) == 0:
                self.logger.info((tid, t))
                self.logger.info('Tag popularity for tag {} is: {}'.format(t, len(tags_total)))
                values_with_pop.append([tid, t, 2])
        cols = list(genome_tags.columns)
        cols.append(TAG_POPULARITY)
        genome_tags = pd.DataFrame(values_with_pop, columns=cols)
        doc_freq = []
        for i, tag in enumerate(genome_tags.values):
            df = genome_scores.loc[genome_scores['tagId'] == tag[0]]
            freq = np.sum(np.array(df["relevance"]) > 0.5)
            if freq == 0 or freq == 1:
                freq = 2
                self.logger.info('Document frequency for tag {} is: zero'.format(tag))
            doc_freq.append(freq)
        genome_tags[DOC_FREQUENCY] = doc_freq
        genome_tags.to_csv(self.tags_info_file, index=False)
        self.logger.info("Done loading the tag popularity and doc frequency for the tags")
        self.weights = np.log(np.array(genome_tags[TAG_POPULARITY])) / np.log(np.array(genome_tags[DOC_FREQUENCY]))

        objects = []
        tags_rel = genome_scores.as_matrix()
        movie_ids = np.unique(np.array(genome_scores[MOVIE_ID]))
        for i, movie_id in enumerate(movie_ids):
            a = i * self.n_features
            b = (i + 1) * self.n_features
            objects.append(tags_rel[a:b, 2])
        objects = np.array(objects)
        movies_df = movies_df[movies_df[MOVIE_ID].isin(movie_ids)]

        for i, tag in enumerate(genome_tags.values):
            movies_df[tag[1]] = objects[:, i]
        movies_df.to_csv(self.movies_file, index=False)

        self.logger.info("Done loading the features for the movies")

        num_of_movies = movie_ids.shape[0]
        combinations_list = np.array(list(combinations(range(num_of_movies), 2)))
        similarity_matrix = dict()

        features = movies_df.as_matrix()[:, 3:]

        for i, j in combinations_list:
            similarity_matrix[get_key_for_indices(i, j)] = weighted_cosine_similarity(self.weights)(features[i],
                                                                                                    features[j])
            self.logger.info("Calculating similarity {},{}, {}".format(i, j,
                                                                       similarity_matrix[get_key_for_indices(i, j)]))

        for i in range(num_of_movies):
            similarity_matrix[get_key_for_indices(i, i)] = 1.0

        series = pd.Series(similarity_matrix)
        matrix_df = pd.DataFrame({'col_major_index': series.index, 'similarity': series.values})
        matrix_df.to_csv(self.similarity_matrix_file, index=False)
        self.logger.info("Done calculating the similarity matrix stored at: {}".format(self.similarity_matrix_file))

    @abstractmethod
    def make_nearest_neighbour_dataset(self, n_instances, n_objects, seed, **kwargs):
        self.logger.info('For instances {} objects {}, seed {}'.format(n_instances, n_objects, seed))
        random_state = check_random_state(seed)
        X = np.empty((n_instances, n_objects, self.n_features), dtype=float)
        scores = np.empty((n_instances, n_objects), dtype=float)
        for i in range(n_instances):
            subset = random_state.choice(self.n_movies, size=n_objects, replace=False)
            X[i] = self.movie_features[subset]
            D = np.ones((n_objects, n_objects), dtype=float)
            for j, k in combinations(np.arange(n_objects), 2):
                D[j, k] = D[k, j] = self.similarity_matrix[get_key_for_indices(subset[j], subset[k])]
            sum_dist = D.mean(axis=0)
            medoid = np.argmax(sum_dist)
            scores[i] = D[medoid]
        return X, scores

    def get_genre_tag_id(self):
        genres = []
        for g in self.movies_df['genres']:
            arr = g.lower().split('|')
            genres.extend(arr)
        genres = list(set(genres))
        genre_tag_id = []
        for row in self.tags_info_df.sort_values('tagpopularity', ascending=False).values:
            if row[1].lower() in genres or len(genre_tag_id) < 18:
                genre_tag_id.append(row[0] - 1)
        return np.array(genre_tag_id)

    @abstractmethod
    def make_critique_fit_dataset(self, n_instances, n_objects, seed, direction, **kwargs):
        self.logger.info('For instances {} objects {}, seed {}, direction {}'.format(n_instances, n_objects, seed,
                                                                                     direction))
        random_state = check_random_state(seed)
        X = []
        scores = []
        length = (int(n_instances / self.n_movies) + 1)
        popular_tags = self.get_genre_tag_id()
        for i, feature in enumerate(self.movie_features):
            if direction == 1:
                quartile_tags = np.where(np.logical_and(feature >= 1 / 3, feature < 2 / 3))[0]
            else:
                quartile_tags = np.where(feature > 1 / 2)[0]
            if len(quartile_tags) < length:
                quartile_tags = popular_tags
            tag_ids = random_state.choice(quartile_tags, size=length)
            distances = [self.similarity_matrix[get_key_for_indices(i, j)] for j in range(self.n_movies)]
            distances = np.array(distances)
            critique_d = critique_dist(feature, self.movie_features, tag_ids, direction=direction, relu=False)
            critique_fit = np.multiply(critique_d, distances)
            orderings = np.argsort(critique_fit, axis=-1)[:, ::-1][:, 0:(n_objects - 1)]
            orderings = np.append(orderings, np.zeros(length, dtype=int)[:, None] + i, axis=1)
            to_remove = []
            for j, ordering in enumerate(orderings):
                random_state.shuffle(ordering)
                if len(np.unique(critique_fit[j, ordering])) != n_objects:
                    to_remove.append(j)
            cf_scores = critique_fit[np.arange(length)[:, None], orderings]
            movies = self.movie_features[orderings]
            if len(to_remove) != 0:
                cf_scores = np.delete(cf_scores, to_remove, 0)
                movies = np.delete(movies, to_remove, 0)
            scores.extend(cf_scores)
            X.extend(movies)
        X = np.array(X)
        scores = np.array(scores)
        indices = random_state.choice(X.shape[0], n_instances, replace=False)
        X = X[indices, :, :]
        scores = scores[indices, :]
        return X, scores

    def get_dataset_dictionaries(self, lengths=[5, 6]):
        x_train = dict()
        y_train = dict()
        x_test = dict()
        y_test = dict()
        for n_obj in lengths:
            seed = self.random_state.randint(2 ** 32, dtype='uint32')
            total_instances = self.n_test_instances + self.n_train_instances
            X, Y = self.dataset_function(total_instances, n_obj, seed=seed)
            x_1, x_2, y_1, y_2 = train_test_split(X, Y, random_state=self.random_state, test_size=self.n_test_instances)
            if self.standardize:
                x_1, x_2 = standardize_features(x_1, x_2)
            x_train[n_obj], x_test[n_obj], y_train[n_obj], y_test[n_obj] = x_1, x_2, y_1, y_2
        self.logger.info('Done')
        return x_train, y_train, x_test, y_test

    def get_train_test_datasets(self, n_datasets=5):
        splits = np.array(n_datasets)
        self.logger.info('Done')
        return self.splitter(splits)

    def get_single_train_test_split(self):
        seed = self.random_state.randint(2 ** 32, dtype='uint32')
        total_instances = self.n_test_instances + self.n_train_instances
        self.X, self.Y = self.dataset_function(total_instances, self.n_objects, seed=seed)
        self.__check_dataset_validity__()
        x_train, x_test, y_train, y_test = train_test_split(self.X, self.Y, random_state=self.random_state,
                                                            test_size=self.n_test_instances)
        if self.standardize:
            x_train, x_test = standardize_features(x_train, x_test)
        self.logger.info('Done')
        return x_train, y_train, x_test, y_test

    def splitter(self, iter):
        for i in iter:
            seed = self.random_state.randint(2 ** 32, dtype='uint32') + i
            total_instances = self.n_test_instances + self.n_train_instances
            X, Y = self.dataset_function(total_instances, self.n_objects, seed=seed)
            x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=self.random_state,
                                                                test_size=self.n_test_instances)
            if self.standardize:
                x_train, x_test = standardize_features(x_train, x_test)

            yield x_train, y_train, x_test, y_test


def weighted_cosine_similarity(weights):
    def distance_function(x, y):
        denominator = np.sqrt(np.sum(weights * x * x)) * np.sqrt(
            np.sum(weights * y * y))
        similarity = np.sum(weights * x * y) / denominator
        return similarity

    return distance_function


def critique_dist(ic, irs, tag_ids, direction=-1, relu=True):
    distances = ((irs[:, tag_ids] - ic[tag_ids]) * direction).T
    if relu:
        distances = np.maximum(np.zeros_like(distances), distances)
    return distances
