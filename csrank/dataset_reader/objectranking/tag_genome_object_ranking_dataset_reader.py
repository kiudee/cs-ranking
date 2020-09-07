import logging

from csrank.constants import OBJECT_RANKING
from csrank.numpy_util import scores_to_rankings
from ..tag_genome_reader import TagGenomeDatasetReader

logger = logging.getLogger(__name__)


class TagGenomeObjectRankingDatasetReader(TagGenomeDatasetReader):
    def __init__(self, dataset_type="similarity", **kwargs):
        super(TagGenomeObjectRankingDatasetReader, self).__init__(
            learning_problem=OBJECT_RANKING, **kwargs
        )
        dataset_func_dict = {
            "nearest_neighbour": self.make_nearest_neighbour_dataset,
            "critique_fit_less": self.make_critique_fit_dataset(direction=-1),
            "critique_fit_more": self.make_critique_fit_dataset(direction=1),
        }
        if dataset_type not in dataset_func_dict:
            raise ValueError(
                f"dataset_type must be one of {set(dataset_func_dict.keys())}"
            )
        logger.info("Dataset type: {}".format(dataset_type))
        self.dataset_function = dataset_func_dict[dataset_type]

    def make_nearest_neighbour_dataset(self, n_instances, n_objects, seed, **kwargs):
        X, scores = super().make_nearest_neighbour_dataset(
            n_instances=n_instances, n_objects=n_objects, seed=seed
        )
        # Higher the similarity lower the rank of the object
        Y = scores_to_rankings(scores)
        return X, Y

    def make_critique_fit_dataset(self, direction):
        def dataset_generator(n_instances, n_objects, seed, **kwargs):
            X, scores = super(
                TagGenomeObjectRankingDatasetReader, self
            ).make_critique_fit_dataset(
                n_instances=n_instances,
                n_objects=n_objects,
                seed=seed,
                direction=direction,
            )
            Y = scores_to_rankings(scores)
            return X, Y

        return dataset_generator
