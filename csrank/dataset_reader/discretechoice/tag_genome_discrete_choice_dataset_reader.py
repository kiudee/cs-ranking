import numpy as np
from sklearn.utils import check_random_state

from csrank.constants import DISCRETE_CHOICE
from csrank.dataset_reader.discretechoice.util import convert_to_label_encoding
from csrank.dataset_reader.tag_genome_reader import critique_dist
from csrank.dataset_reader.util import get_key_for_indices
from ..tag_genome_reader import TagGenomeDatasetReader


class TagGenomeDiscreteChoiceDatasetReader(TagGenomeDatasetReader):
    def __init__(self, dataset_type="similarity", **kwargs):
        super(TagGenomeDiscreteChoiceDatasetReader, self).__init__(learning_problem=DISCRETE_CHOICE, **kwargs)
        dataset_func_dict = {"nearest_neighbour": self.make_nearest_neighbour_dataset,
                             "critique_fit_less": self.make_critique_fit_dataset(direction=-1),
                             "critique_fit_more": self.make_critique_fit_dataset(direction=1),
                             "dissimilar_nearest_neighbour": self.make_dissimilar_nearest_neighbour_dataset,
                             "dissimilar_critique_more": self.make_dissimilar_critique_dataset(direction=1),
                             "dissimilar_critique_less": self.make_dissimilar_critique_dataset(direction=-1)}
        if dataset_type not in dataset_func_dict.keys():
            dataset_type = "nearest_neighbour"
        self.logger.info('Dataset type: {}'.format(dataset_type))
        self.dataset_function = dataset_func_dict[dataset_type]

    def make_nearest_neighbour_dataset(self, n_instances, n_objects, seed, **kwargs):
        X, scores = super(TagGenomeDiscreteChoiceDatasetReader, self).make_nearest_neighbour_dataset(
            n_instances=n_instances, n_objects=n_objects, seed=seed)
        # Higher the similarity lower the rank of the object, getting the object with second highest similarity
        Y = np.argsort(scores, axis=1)[:, -2]
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
        X, scores = super(TagGenomeDiscreteChoiceDatasetReader, self).make_nearest_neighbour_dataset(
            n_instances=n_instances, n_objects=n_objects, seed=seed)
        # Higher the similarity lower the rank of the object, getting the object with second highest similarity
        Y = np.argsort(scores, axis=1)[:, 0]
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
                critique_d = critique_dist(feature, self.movie_features, tag_ids, direction=direction, relu=False)
                critique_fit = np.multiply(critique_d, distances)
                orderings = np.argsort(critique_fit, axis=-1)[:, ::-1]
                minimum = np.zeros(length, dtype=int)
                for k, dist in enumerate(critique_fit):
                    quartile = np.percentile(dist, [0, 5])
                    last = np.where(np.logical_and((dist >= quartile[0]), (dist <= quartile[1])))[0]
                    if i in last:
                        index = np.where(last == i)[0][0]
                        last = np.delete(last, index)
                    minimum[k] = random_state.choice(last, size=1)[0]
                orderings = orderings[:, 0:n_objects - 2]
                orderings = np.append(orderings, minimum[:, None], axis=1)
                orderings = np.append(orderings, np.zeros(length, dtype=int)[:, None] + i, axis=1)
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
