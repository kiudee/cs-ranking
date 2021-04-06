"""Apply an evaluation function to instances and collect the results.

The modules that are defined here should take a tensor of instances (shape
:math:`(N, O, F)`, where :math:`N` is the number of instances, :math:`O` the
number of objects per instance and :math:`F` the number of features per object)
and apply some order :math:`k` utility function on all possible combinations of
:math:`k` objects per instance. It should return an evaluation result matrix of
shape :math:`(N, O^k)`.

As an example, consider a pairwise preference function used together with
``PairwiseEvaluation``. If you feed ``PairwiseEvaluation`` with a tensor of
instances, you get back a tensor of preference matrices where the entry at
indes :math:`(i, j)` tells you how preferable object :math:`i` is to object
:math:`j`. You can then combine this result with a reduction to get a
first-order (or order-:math:`k`) utility module. This is used in the
``MeanAggregatedUtility`` first-order utility.
"""

import torch
import torch.nn as nn


def _generate_pairs(instances):
    """Generate object pairs from instances.

    This can be used to generate the individual comparisons in a first-order
    utility function model.

    >>> object_a = [0.5, 0.6]
    >>> object_b = [1.5, 1.6]
    >>> object_c = [2.5, 2.6]
    >>> object_d = [3.5, 3.6]
    >>> object_e = [4.5, 4.6]
    >>> object_f = [5.5, 5.6]
    >>> # instance = list of objects to rank
    >>> instance_a = [object_a, object_b, object_c]
    >>> instance_b = [object_d, object_e, object_f]
    >>> instances = [instance_a, instance_b]

    >>> _generate_pairs(torch.tensor(instances))
    tensor([[[0.5000, 0.6000, 0.5000, 0.6000],
             [0.5000, 0.6000, 1.5000, 1.6000],
             [0.5000, 0.6000, 2.5000, 2.6000],
             [1.5000, 1.6000, 0.5000, 0.6000],
             [1.5000, 1.6000, 1.5000, 1.6000],
             [1.5000, 1.6000, 2.5000, 2.6000],
             [2.5000, 2.6000, 0.5000, 0.6000],
             [2.5000, 2.6000, 1.5000, 1.6000],
             [2.5000, 2.6000, 2.5000, 2.6000]],
    <BLANKLINE>
            [[3.5000, 3.6000, 3.5000, 3.6000],
             [3.5000, 3.6000, 4.5000, 4.6000],
             [3.5000, 3.6000, 5.5000, 5.6000],
             [4.5000, 4.6000, 3.5000, 3.6000],
             [4.5000, 4.6000, 4.5000, 4.6000],
             [4.5000, 4.6000, 5.5000, 5.6000],
             [5.5000, 5.6000, 3.5000, 3.6000],
             [5.5000, 5.6000, 4.5000, 4.6000],
             [5.5000, 5.6000, 5.5000, 5.6000]]])
    """

    def repeat_individual_objects(instances, times):
        """Repeat each object once, immediately after the original."""
        # add a dimension, so that each object is now enclosed in a singleton
        unsqueezed = instances.unsqueeze(2)

        # repeat every object (along the newly added dimension)
        # ([[object_a], [object_a]], [[object_b]], [[object_b]])
        repeated = unsqueezed.repeat(1, 1, times, 1)
        # collapse the added dimension again so that each object is on the same
        # level ([object_a], [object_a], [object_b]], [[object_b])
        return repeated.view(instances.size(0), -1, instances.size(2))

    def repeat_object_lists(instances, times):
        """Repeat the whole object list as a unit (the same as "first" but in a different order)."""
        return instances.repeat(1, times, 1)

    n_objects = instances.size(1)
    first = repeat_individual_objects(instances, n_objects)
    second = repeat_object_lists(instances, n_objects)

    # Glue the two together at the object level (the object's feature vectors
    # are concatenated)
    output_tensor = torch.cat((first, second), dim=2)
    return output_tensor


class PairwiseEvaluation(nn.Module):
    """A module that applies a first-order utility to all object pairs.

    This module receives a list of instances as an input:

    >>> object_00 = [0.5, 0.6]
    >>> object_01 = [1.5, 1.6]
    >>> object_02 = [2.5, 2.6]
    >>> object_10 = [3.5, 3.6]
    >>> object_11 = [4.5, 4.6]
    >>> object_12 = [5.5, 5.6]
    >>> # instance = list of objects to rank
    >>> instance_a = [object_00, object_01, object_02]
    >>> instance_b = [object_10, object_11, object_12]
    >>> instances = torch.tensor([instance_a, instance_b])
    >>> n_features = instances.size(2)

    And computes a list of *relations*, one relation for each instance. A
    relation is a 2d array that contains a score for each pairwise object
    combination.

    >>> import numpy as np
    >>> from csrank.modules.object_mapping import DeterministicSumming
    >>> utility = PairwiseEvaluation(DeterministicSumming(2 * n_features))
    >>> utilities = utility(torch.tensor(instances))
    >>> utilities
    tensor([[[ 2.2000,  4.2000,  6.2000],
             [ 4.2000,  6.2000,  8.2000],
             [ 6.2000,  8.2000, 10.2000]],
    <BLANKLINE>
            [[14.2000, 16.2000, 18.2000],
             [16.2000, 18.2000, 20.2000],
             [18.2000, 20.2000, 22.2000]]])
    >>> utilities[0][0][1] == np.sum(object_00) + np.sum(object_01)
    tensor(True)
    """

    def __init__(self, pairwise_model):
        super().__init__()
        self.pairwise_model = pairwise_model

    def forward(self, instances):
        n_instances = instances.size(0)
        n_objects = instances.size(1)
        object_pairs = _generate_pairs(instances)
        scores = self.pairwise_model(object_pairs)
        return scores.view(n_instances, n_objects, n_objects)
