from itertools import product

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state

from csrank.constants import OBJECT_RANKING
from csrank.dataset_reader.util import get_key_for_indices, weighted_cosine_distance
from csrank.util import scores_to_rankings
from ..tag_genome_reader import TagGenomeDatasetReader


class TagGenomeObjectRankingDatasetReader(TagGenomeDatasetReader):
    def __init__(self, **kwargs):
        super(TagGenomeObjectRankingDatasetReader, self).__init__(learning_problem=OBJECT_RANKING, **kwargs)

    def make_similarity_based_dataset(self, n_instances, n_objects, seed=42):

        random_state = check_random_state(seed)
        X = np.empty((n_instances, n_objects, self.n_features), dtype=float)
        similarity_scores = np.empty((n_instances, n_objects), dtype=float)
        movie_features = self.movies_df.as_matrix()[:, 3:]
        for i in range(n_instances):
            subset = random_state.choice(movie_features.shape[0], size=n_objects, replace=False)
            X[i] = movie_features[subset]
            query = random_state.choice(n_objects, size=1)
            one_row = [self.similarity_matrix_lin_list[get_key_for_indices(i, j)] for i, j in
                       product(subset[query], subset)]
            similarity_scores[i] = 1 - np.array(one_row)

        # Higher the similarity lower the rank of the object
        rankings = scores_to_rankings(similarity_scores)
        return X, rankings

    def make_nearest_neighbour_dataset(self, n_instances, n_objects, seed):
        random_state = check_random_state(seed)
        length = (int(n_instances / self.n_movies) + 1)
        n_neighbors = length * n_objects
        dist_func = weighted_cosine_distance(self.weights)
        self.model = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto', metric=dist_func)
        movie_features = self.movies_df.as_matrix()[:, 3:]
        self.model.fit(movie_features)
        distances, set_of_movies = self.model.kneighbors(movie_features, n_neighbors, return_distance=True)
        similarity_scores = 1.0 - distances
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
        rankings = scores_to_rankings(scores)
        return X, rankings
