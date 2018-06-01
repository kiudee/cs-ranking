from csrank.constants import DISCRETE_CHOICE
from csrank.dataset_reader.discretechoice.util import convert_to_label_encoding
from ..tag_genome_reader import TagGenomeDatasetReader


class TagGenomeDiscreteChoiceDatasetReader(TagGenomeDatasetReader):
    def __init__(self, dataset_type="similarity", **kwargs):
        super(TagGenomeDiscreteChoiceDatasetReader, self).__init__(learning_problem=DISCRETE_CHOICE, **kwargs)
        dataset_func_dict = {"similarity": self.make_similarity_based_dataset,
                             "nearest_neighbour": self.make_nearest_neighbour_dataset,
                             "critique_fit_less": self.make_critique_fit_dataset(direction=-1),
                             "critique_fit_more": self.make_critique_fit_dataset(direction=1)}
        if dataset_type not in dataset_func_dict.keys():
            dataset_type = "similarity"

        self.dataset_function = dataset_func_dict[dataset_type]

    def make_similarity_based_dataset(self, n_instances, n_objects, seed, **kwargs):
        X, scores = super().make_similarity_based_dataset(n_instances=n_instances, n_objects=n_objects, seed=seed)
        # Higher the similarity lower the rank of the object
        Y = scores.argmax(axis=1)
        Y = convert_to_label_encoding(Y, n_objects)
        return X, Y

    def make_nearest_neighbour_dataset(self, n_instances, n_objects, seed, **kwargs):
        X, scores = super().make_nearest_neighbour_dataset(n_instances=n_instances, n_objects=n_objects, seed=seed)
        # Higher the similarity lower the rank of the object
        Y = scores.argmax(axis=1)
        Y = convert_to_label_encoding(Y, n_objects)
        return X, Y

    def make_critique_fit_dataset(self, direction):
        def dataset_generator(n_instances, n_objects, seed, **kwargs):
            X, scores = super().make_critique_fit_dataset(direction)(n_instances=n_instances, n_objects=n_objects,
                                                                     seed=seed)
            Y = scores.argmax(axis=1)
            Y = convert_to_label_encoding(Y, n_objects)
            return X, Y

        return dataset_generator
