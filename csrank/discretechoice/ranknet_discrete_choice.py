import functools

import torch.nn as nn

from csrank.discrete_choice_losses import CategoricalHingeLossMax
from csrank.discretechoice.discrete_choice import SkorchDiscreteChoiceFunction
from csrank.modules.object_mapping import DenseNeuralNetwork
from csrank.modules.scoring.ranknet import RankNetScoring


class RankNetDiscreteChoiceFunction(SkorchDiscreteChoiceFunction):
    """A discrete choice estimator based on the RankNet-Approach.

    See the documentation of :class:`csrank.modules.scoring.RankNetScoring` for
    more details.

    Parameters
    ----------
    n_hidden : int
        The number of hidden layers that should be used in the utility module.

    n_units : int
        The number of units per hidden layer that should be used in the utility
        module.

    choice_size : int
        The size of the target choice set.

    activation : torch activation function (class)
        The activation function that should be used for each layer of the
        comparative network.

    criterion : torch criterion (class)
        The criterion that is used to evaluate and optimize the module.

    **kwargs : skorch NeuralNet arguments
        All keyword arguments are passed to the constructor of
        ``SkorchDiscreteChoiceFunction``. See the documentation of that class
        for more details.
    """

    def __init__(
        self,
        n_hidden=2,
        n_units=8,
        activation=nn.ReLU,
        choice_size=1,
        criterion=CategoricalHingeLossMax,
        **kwargs
    ):
        self.n_hidden = n_hidden
        self.n_units = n_units
        self.activation = activation
        super().__init__(
            module=RankNetScoring,
            criterion=criterion,
            choice_size=choice_size,
            **kwargs
        )

    def _get_extra_module_parameters(self):
        """Return extra parameters that should be passed to the module."""
        params = super()._get_extra_module_parameters()
        params["object_utility_module"] = functools.partial(
            DenseNeuralNetwork,
            hidden_layers=self.n_hidden,
            units_per_hidden=self.n_units,
            activation=self.activation(),
            output_size=1,
        )
        return params
