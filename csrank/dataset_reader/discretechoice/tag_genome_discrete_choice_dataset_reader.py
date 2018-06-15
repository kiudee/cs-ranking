import numpy as np
from sklearn.utils import check_random_state

from csrank.constants import DISCRETE_CHOICE
from csrank.dataset_reader.discretechoice.util import convert_to_label_encoding
from csrank.dataset_reader.util import get_key_for_indices, critique_dist
from ..tag_genome_reader import TagGenomeDatasetReader


class TagGenomeDiscreteChoiceDatasetReader(TagGenomeDatasetReader):
    def __init__(self, dataset_type="similarity", **kwargs):
        super(TagGenomeDiscreteChoiceDatasetReader, self).__init__(learning_problem=DISCRETE_CHOICE, **kwargs)
        dataset_func_dict = {"similarity": self.make_similarity_based_dataset,
                             "nearest_neighbour": self.make_nearest_neighbour_dataset,
                             "critique_fit_less": self.make_critique_fit_dataset(direction=-1),
                             "critique_fit_more": self.make_critique_fit_dataset(direction=1),
                             "dissimilar_nearest_neighbour": self.make_dissimilar_nearest_neighbour_dataset,
                             "dissimilar_critique_more": self.make_dissimilar_critique_dataset(direction=1),
                             "dissimilar_critique_less": self.make_dissimilar_critique_dataset(direction=-1)}
        if dataset_type not in dataset_func_dict.keys():
            dataset_type = "similarity"

        self.dataset_function = dataset_func_dict[dataset_type]

    def make_similarity_based_dataset(self, n_instances, n_objects, seed, **kwargs):
        X, scores = super(TagGenomeDiscreteChoiceDatasetReader, self).make_similarity_based_dataset(
            n_instances=n_instances, n_objects=n_objects, seed=seed)
        # Higher the similarity lower the rank of the object
        Y = scores.argmax(axis=1)
        Y = convert_to_label_encoding(Y, n_objects)
        return X, Y

    def make_nearest_neighbour_dataset(self, n_instances, n_objects, seed, **kwargs):
        X, scores = super(TagGenomeDiscreteChoiceDatasetReader, self).make_nearest_neighbour_dataset(
            n_instances=n_instances, n_objects=n_objects, seed=seed)
        # Higher the similarity lower the rank of the object
        Y = scores.argmax(axis=1)
        Y = convert_to_label_encoding(Y, n_objects)
        return X, Y

    def make_critique_fit_dataset(self, direction):
        def dataset_generator(n_instances, n_objects, seed, **kwargs):
            X, scores = super(TagGenomeDiscreteChoiceDatasetReader, self).make_critique_fit_dataset(
                n_instances=n_instances, n_objects=n_objects, seed=seed, direction=direction)
            Y = scores.argmax(axis=1)
            Y = convert_to_label_encoding(Y, n_objects)
            return X, Y

        return dataset_generator

    def make_dissimilar_nearest_neighbour_dataset(self, n_instances, n_objects, seed, **kwargs):
        self.logger.info('For instances {} objects {}, seed {}'.format(n_instances, n_objects, seed))
        random_state = check_random_state(seed)
        length = (int(n_instances / self.n_movies) + 1)
        X = []
        scores = []
        for i, feature in enumerate(self.movie_features):
            distances = [self.similarity_matrix[get_key_for_indices(i, j)] for j in range(self.n_movies)]
            distances = np.array(distances)
            orderings = np.argsort(distances)[::-1]
            quartile = np.percentile(distances, [95, 97.5])
            last = np.where(np.logical_and(distances >= quartile[0], distances <= quartile[1]))[0]
            minimum = random_state.choice(last, size=length, replace=False)[:, None]

            # last = np.min([1000, self.n_movies - 50 * n_objects * length])
            # minimum = random_state.choice(last, size=length, replace=False)
            # minimum = orderings[-minimum][:, None]
            orderings = orderings[1:((n_objects - 1) * length + 1)]
            orderings = orderings.reshape(length, n_objects - 1)
            orderings = np.append(orderings, minimum, axis=1)
            for o in orderings:
                random_state.shuffle(o)
            scores.extend(distances[orderings])
            X.extend(self.movie_features[orderings])
        X = np.array(X)
        scores = np.array(scores)
        indices = random_state.choice(X.shape[0], n_instances, replace=False)
        X = X[indices, :, :]
        scores = scores[indices, :]
        Y = scores.argmin(axis=1)
        Y = convert_to_label_encoding(Y, n_objects)
        return X, Y

    def make_dissimilar_critique_dataset(self, direction):
        def dataset_generator(n_instances, n_objects, seed, **kwargs):
            self.logger.info('For instances {} objects {}, seed {}, direction {}'.format(n_instances, n_objects, seed,
                                                                                         direction))
            random_state = check_random_state(seed)
            X = []
            scores = []
            length = (int(n_instances / self.n_movies) + 1)
            for i, feature in enumerate(self.movie_features):
                quartile = np.percentile(feature, [95, 97.5])
                tag_ids = np.where(np.logical_and((feature > quartile[0]), (feature < quartile[1])))[0]
                tag_ids = tag_ids[np.argsort(feature[tag_ids])[::-1]]
                if direction == -1:
                    tag_ids = tag_ids[0:length]
                else:
                    tag_ids = tag_ids[-length:]
                distances = [self.similarity_matrix[get_key_for_indices(i, j)] for j in range(self.n_movies)]
                distances = np.array(distances) - 0.61
                critique_d = critique_dist(feature, self.movie_features, tag_ids, direction=direction)
                critique_fit = np.multiply(critique_d, distances)
                orderings = np.argsort(critique_fit, axis=-1)[:, ::-1]
                minimum = np.zeros(length, dtype=int)
                k = 0
                for dist in critique_fit:
                    last = np.where(dist == 0)[0]
                    index = np.where(last == i)[0][0]
                    last = np.delete(last, index)
                    minimum[k] = random_state.choice(last, size=1)[0]
                    k += 1
                orderings = orderings[:, 0:n_objects - 1]
                orderings = np.append(orderings, minimum[:, None], axis=1)
                for o in orderings:
                    random_state.shuffle(o)
                scores.extend(critique_fit[np.arange(length)[:, None], orderings])
                X.extend(self.movie_features[orderings])
            X = np.array(X)
            scores = np.array(scores)
            indices = random_state.choice(X.shape[0], n_instances, replace=False)
            X = X[indices, :, :]
            scores = scores[indices, :]

            Y = scores.argmin(axis=1)
            Y = convert_to_label_encoding(Y, n_objects)
            return X, Y

        return dataset_generator
