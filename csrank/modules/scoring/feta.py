"""An implementation of the scoring module for FATE estimators."""

import functools

import torch.nn as nn

from csrank.modules.collection import PairwiseEvaluation
from csrank.modules.instance_reduction import MeanAggregatedUtility
from csrank.modules.object_mapping import DenseNeuralNetwork


class FETAScoring(nn.Module):
    """Map instances to scores with the FETA approach.

    >>> from csrank.modules.object_mapping import DeterministicSumming
    >>> scoring = FETAScoring(
    ...     n_features=2,
    ...     pairwise_utility_module=DeterministicSumming,
    ... )

    Now let's define some problem instances.

    >>> object_a = [0.5, 0.8]
    >>> object_b = [1.5, 1.8]
    >>> object_c = [2.5, 2.8]
    >>> object_d = [3.5, 3.6]
    >>> object_e = [4.5, 4.6]
    >>> object_f = [5.5, 5.6]
    >>> # instance = list of objects to rank
    >>> instance_a = [object_a, object_b, object_c]
    >>> instance_b = [object_d, object_e, object_f]
    >>> import torch
    >>> instances = torch.tensor([instance_a, instance_b])

    >>> scoring(instances)
    tensor([[ 5.6000,  6.6000,  7.6000],
            [17.2000, 18.2000, 19.2000]])

    Parameters
    ----------
    n_features: int
        The number of features each object has.

    pairwise_utility_module: pytorch module with one integer parameter
        The module that should be used for pairwise utility estimations. Uses a
        simple linear mapping if not specified. You likely want to replace this
        with something more expressive such as a ``DenseNeuralNetwork``. This
        should take the size of the input values as its only parameter. You can
        use ``functools.partial`` if necessary. This corresponds to :math:`U`
        [1]_.

    zeroth_order_module: pytorch module with one integer parameter
        The module that should be used to evaluate objects in isolation. May be
        ``None``, in which case no zeroth order module is used. That is the
        default behavior. You may want to replace this with something like a
        ``DenseNeuralNetwork``. This should take the size of the input values
        as its only parameter. You can use ``functools.partial`` if necessary.
        This corresponds to :math:`U` This corresponds to :math:`U_0` [1]_.
    """

    def __init__(
        self,
        n_features,
        pairwise_utility_module=functools.partial(
            DenseNeuralNetwork, hidden_layers=3, units_per_hidden=20
        ),
        zeroth_order_module=None,
    ):
        super().__init__()
        # Compute the utility of each object in the context of every other
        # object. Use mean aggregation to come to a single utility value per
        # object. Ignore self-comparisons. Including those would be similar to
        # adding a zeroth-order model, but with a different scale and learned
        # included in the same utility network. That could make it harder to
        # learn.
        self.mean_aggregated_utilty = MeanAggregatedUtility(
            PairwiseEvaluation(pairwise_utility_module(2 * n_features)),
            exclude_self_comparison=True,
        )
        self.zeroth_order_module = (
            None if zeroth_order_module is None else zeroth_order_module(n_features)
        )

    def forward(self, instances, **kwargs):
        total_utility = self.mean_aggregated_utilty(instances)
        if self.zeroth_order_module is not None:
            # The result of the zeroth order utility module has a singleton
            # dimension, while the aggregated utility does not.
            zeroth_order_utility = self.zeroth_order_module(instances).reshape(
                total_utility.shape
            )
            total_utility += zeroth_order_utility
        return total_utility
