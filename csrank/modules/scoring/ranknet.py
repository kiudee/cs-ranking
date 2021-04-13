"""An implementation of the scoring module for CmpNet estimators."""

import functools

import torch.nn as nn

from csrank.modules.object_mapping import DenseNeuralNetwork


class RankNetScoring(nn.Module):
    """Map instances to scores with the RankNet approach.

    This approach learns a utility function that considers each object in
    isolation, without taking context into account. It is similar to the
    approach that was introduced in [1]_. This version trains the network based
    on a loss function that evaluates the result (general/discrete choice or
    ranking) as a whole. This differs from the version in [1]_, which proposes
    to evaluate utilities on object pairs and train the network based on that.

    Parameters
    ----------
    n_features: int
        The number of features each object has.

    object_utility_module: pytorch module with one integer parameter
        The module that should be used for utility estimations. Uses a simple
        linear mapping if not specified. You likely want to replace this with
        something more expressive such as a ``DenseNeuralNetwork``. This should
        take the size of the input values as its only parameter. You can use
        ``functools.partial`` if necessary.

    References
    ----------
    .. [1] Burges, C., Shaked, T., Renshaw, E., Lazier, A., Deeds, M.,
    Hamilton, N., & Hullender, G. (2005, August). Learning to rank using
    gradient descent. In Proceedings of the 22nd international conference on
    Machine learning (pp. 89-96).
    """

    def __init__(
        self,
        n_features,
        object_utility_module=functools.partial(
            DenseNeuralNetwork,
            hidden_layers=3,
            units_per_hidden=20,
            ouput_features=1,
        ),
    ):
        super().__init__()
        self.object_utility_module = object_utility_module(n_features)

    def forward(self, instances):
        return self.object_utility_module(instances).squeeze(dim=-1)
