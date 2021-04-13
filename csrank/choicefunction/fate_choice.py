import functools

import torch.nn as nn

from csrank.choicefunction.choice_functions import SkorchChoiceFunction
from csrank.modules.object_mapping import DenseNeuralNetwork
from csrank.modules.scoring import FATEScoring


class FATEChoiceFunction(SkorchChoiceFunction):
    """A variable choice estimator based on the FATE-Approach.

    See the documentation of :class:`csrank.modules.scoring.FATEScoring` for
    more details.

    Parameters
    ----------
    n_hidden_set_layers : int
        The number of hidden layers that should be used for the ``DeepSet``
        context embedding.

    n_hidden_set_untis : int
        The number of units per hidden layer that should be used for the
        ``DeepSet`` context embedding.

    n_hidden_joint_layers : int
        The number of hidden layers that should be used for the utility
        function that evaluates each object in the aggregated context.

    n_hidden_joint_units : int
        The number of units per hidden layer that should used for the utility
        function that evaluates each object in the aggregated context.

    activation : torch activation function (class)
        The activation function that should be used for each layer of the two
        ("set" and "joint) neural networks.

    criterion : torch criterion (class)
        The criterion that is used to evaluate and optimize the module.

    **kwargs : skorch NeuralNet arguments
        All keyword arguments are passed to the constructor of
        ``SkorchChoiceFunction``. See the documentation of that class for more
        details.
    """

    def __init__(
        self,
        n_hidden_set_layers=2,
        n_hidden_set_units=32,
        n_hidden_joint_layers=2,
        n_hidden_joint_units=32,
        activation=nn.SELU,
        criterion=nn.BCELoss,
        **kwargs
    ):
        self.n_hidden_set_layers = n_hidden_set_layers
        self.n_hidden_set_units = n_hidden_set_units
        self.n_hidden_joint_layers = n_hidden_joint_layers
        self.n_hidden_joint_units = n_hidden_joint_units
        self.activation = activation
        super().__init__(module=FATEScoring, criterion=criterion, **kwargs)

    def _get_extra_module_parameters(self):
        """Return extra parameters that should be passed to the module."""
        params = super()._get_extra_module_parameters()
        params["pairwise_utility_module"] = functools.partial(
            DenseNeuralNetwork,
            hidden_layers=self.n_hidden_joint_layers,
            units_per_hidden=self.n_hidden_joint_units,
            activation=self.activation(),
            output_size=1,
        )
        params["embedding_module"] = functools.partial(
            DenseNeuralNetwork,
            hidden_layers=self.n_hidden_set_layers,
            units_per_hidden=self.n_hidden_set_units,
            activation=self.activation(),
        )
        return params
