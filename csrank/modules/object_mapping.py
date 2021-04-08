"""Modules that transform feature vectors individually.

The modules listed here should take a 1 or higher dimensional input and apply
some mapping to the last dimension.  They do not take interactions of different
feature vectors into account.

Inputs of shape :math:`(N, *, H_i)` are transformed to outputs of shape
:math:`(N, *, H_o)`. In this case :math:`N` is the batch size, :math:`*`
denotes arbitrary additional dimension (which are preserved), :math:`H_i`
refers to the number of input features :math:`H_o` refers to the number of
input and output features respectively.
"""

import torch.nn as nn

# Refer to Figure 1 of https://arxiv.org/pdf/1901.10860.pdf for an overview of
# the different components that are used for FATE and FETA.


class DeterministicSumming(nn.Module):
    """Transform a tensor into repetitions of its sum.

    Intended for use in tests, not useful for actual learning. The last
    dimension of the input should contain feature vectors. The result will be
    an array of matching shape with the last dimension replaced by repeated
    utility values (i.e. sums).

    Let's use this as a pairwise utility function. As an example, consider
    this pairing. There are two instances with two objects each. All object
    combinations are considered. Objects have two features.

    >>> import torch
    >>> pairs = torch.tensor(
    ...    [[[0.5000, 0.6000, 0.5000, 0.6000],
    ...      [0.5000, 0.6000, 1.5000, 1.6000],
    ...      [1.5000, 1.6000, 0.5000, 0.6000],
    ...      [1.5000, 1.6000, 1.5000, 1.6000]],
    ...     [[2.5000, 2.6000, 2.5000, 2.6000],
    ...      [2.5000, 2.6000, 3.5000, 3.6000],
    ...      [3.5000, 3.6000, 2.5000, 2.6000],
    ...      [3.5000, 3.6000, 3.5000, 3.6000]]])

    We can compute the mock utility of this pairing as follows:

    >>> utility = DeterministicSumming(input_size=2)
    >>> utilities = utility(pairs)
    >>> utilities
    tensor([[[ 2.2000],
             [ 4.2000],
             [ 4.2000],
             [ 6.2000]],
    <BLANKLINE>
            [[10.2000],
             [12.2000],
             [12.2000],
             [14.2000]]])

    Note that for example :math:`2.2 = 0.5 + 0.6 + 0.5 + 0.6`, that is

    >>> utilities[0][0] == pairs[0][0].sum()
    tensor([True])

    Parameters
    ----------
    input_size : int
        The size of the last dimension of the input.

    output_size : int
        The size of the last dimension of the output. Defaults to `1` to make
        it more convenient to use this as a utility.
    """

    def __init__(self, input_size: int, output_size: int = 1):
        super().__init__()
        self.output_size = output_size

    def forward(self, inputs):
        """Forward inputs through the network.

        Parameters
        ----------
        inputs : tensor
            The input tensor of shape (N, *, I), where I is the input size.

        Returns
        -------
        tensor
            A tensor of shape (N, *, O), where O is the output size.
        """
        summed = inputs.sum(dim=-1)
        # repeat in newly created last dimension
        repeated = (
            summed.view(-1, 1)
            .repeat(1, self.output_size)
            .view(summed.shape + (self.output_size,))
        )
        return repeated


class DenseNeuralNetwork(nn.Module):
    """Deep, densely connected neural network.

    All hidden layers have the same number of units.

    Parameters
    ----------
    input_size: int
        The number of units at the input layer.
    output_size: int
        The number of units at the output layer.
    hidden_layers: int
        The number of hidden layers in addition to the input and output layer.
    units_per_hidden: int
        The number of units each hidden layer has.
    activation: torch activation function
        The activation function that should be applied after each layer.
        Defaults to an instance of `nn.SELU`.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_layers: int,
        units_per_hidden: int,
        activation=None,
    ):
        super().__init__()
        self.input_layer = nn.Linear(input_size, units_per_hidden)
        # ModuleList is necessary to make pytorch aware of these layers and add
        # their parameters to this module's parameter list.
        self.hidden_layers = nn.ModuleList(
            [
                nn.Linear(units_per_hidden, units_per_hidden)
                for _ in range(hidden_layers)
            ]
        )
        self.output_layer = nn.Linear(units_per_hidden, output_size)
        self.activation = activation if activation is not None else nn.SELU()

    def forward(self, x):
        """Forward inputs through the network.

        Parameters
        ----------
        inputs : tensor
            The input tensor of shape (N, *, I), where I is the input size.

        Returns
        -------
        tensor
            A tensor of shape (N, *, O), where O is the output size.
        """
        result = self.activation(self.input_layer(x))
        for layer in self.hidden_layers:
            result = self.activation(layer(result))
        result = self.output_layer(result)
        return result
