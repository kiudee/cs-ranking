from itertools import product

import numpy as np
from sklearn.preprocessing import StandardScaler

from csrank.constants import OBJECT_RANKING
from csrank.dataset_reader.util import get_key_for_indices
from csrank.util import scores_to_rankings
from ..tag_genome_reader import TagGenomeDatasetReader

TAG_POPULARITY = "TagPopularity"
DOC_FREQUENCY = "DocFrequency"


class TagGenomeObjectRankingDatasetReader(TagGenomeDatasetReader):
    def __init__(self, **kwargs):
        super(TagGenomeObjectRankingDatasetReader, self).__init__(learning_problem=OBJECT_RANKING, **kwargs)

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
            similarity_scores[i] = 1.0 - np.array(one_row)

        # Higher the similarity lower the rank of the object
        rankings = scores_to_rankings(similarity_scores)

        for i, x in enumerate(X):
            x = StandardScaler().fit_transform(x)
            X[i] = x
        return X, rankings

    def make_nearest_neighbour_dataset(self, n_instances, seed):
        pass

    def splitter(self, iter):
        for i in iter:
            seed = self.random_state.randint(2 ** 32, dtype='uint32') + i
            X_train, Y_train = self.dataset_function(self.n_train_instances, seed=seed)
            X_test, Y_test = self.dataset_function(self.n_test_instances, seed=seed + 1)
        yield X_train, Y_train, X_test, Y_test

    def get_dataset_dictionaries(self):
        pass

    def get_train_test_datasets(self, n_datasets=5):
        splits = np.array(n_datasets)
        return self.splitter(splits)

    def get_single_train_test_split(self):
        seed = self.random_state.randint(2 ** 32, dtype='uint32')
        self.X, self.Y = X_train, Y_train = self.dataset_function(self.n_train_instances, seed=seed)
        self.__check_dataset_validity__()

        self.X, self.Y = X_test, Y_test = self.dataset_function(self.n_test_instances, seed=seed + 1)
        self.__check_dataset_validity__()
        return X_train, Y_train, X_test, Y_test
