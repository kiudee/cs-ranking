import functools

import torch.nn as nn

from csrank.discrete_choice_losses import CategoricalHingeLossMax
from csrank.discretechoice.discrete_choice import SkorchDiscreteChoiceFunction
from csrank.modules.object_mapping import DenseNeuralNetwork
from csrank.modules.scoring.cmpnet import CmpNetScoring


class CmpNetDiscreteChoiceFunction(SkorchDiscreteChoiceFunction):
    """A discrete choice estimator based on the CmpNet-Approach.

    See the documentation of :class:`csrank.modules.scoring.CmpNetScoring` for
    more details.

    Parameters
    ----------
    n_hidden : int
        The number of hidden layers that should be used in the pairwise utility
        module and the zeroth order module (if enabled).

    n_units : int
        The number of units per hidden layer that should be used in the
        pairwise utility module and the zeroth order module (if enabled).

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
        criterion=CategoricalHingeLossMax,
        activation=nn.SELU,
        choice_size=1,
        **kwargs
    ):
        self.n_hidden = n_hidden
        self.n_units = n_units
        self.activation = activation
        super().__init__(
            module=CmpNetScoring, criterion=criterion, choice_size=choice_size, **kwargs
        )

    def _get_extra_module_parameters(self):
        """Return extra parameters that should be passed to the module."""
        params = super()._get_extra_module_parameters()
        params["pairwise_preference_core"] = functools.partial(
            DenseNeuralNetwork,
            hidden_layers=self.n_hidden,
            units_per_hidden=self.n_units,
            activation=self.activation(),
        )
        params["core_encoding_size"] = self.n_hidden
        return params
