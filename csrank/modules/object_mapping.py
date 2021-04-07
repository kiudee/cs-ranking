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

import torch
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


def _flip_pairs(pairs):
    """Flip the order of concatenated feature vectors.

    Given a tensor of concatenated feature vectors, such as this one

    >>> pairs = torch.tensor([
    ...     [1,  2,  3,  4],
    ...     [5,  6,  7,  8],
    ...     [9, 10, 11, 12],
    ... ])

    this function will return a tensor of concatenated feature vectors where
    the order of the objects is reversed:

    >>> _flip_pairs(pairs)
    tensor([[ 3,  4,  1,  2],
            [ 7,  8,  5,  6],
            [11, 12,  9, 10]])

    Parameters
    ----------
    pairs: tensor of dimension N >= 1
        The tensor of pairs. It is assumed that the last dimension consists of
        concatenated feature vectors of equal length, i.e. that its size is
        even.

    Returns
    -------
    tensor: of dimension N
        The tensor of pairs with the order of the feature vectors reversed.
    """
    previous_shape = pairs.shape
    assert previous_shape[-1] % 2 == 0
    # split concatenated objects
    new_shape = previous_shape[:-1] + (2, previous_shape[-1] // 2)
    split_pairs = pairs.view(*new_shape)

    # reverse the order of the objects
    reverse_split_pairs = split_pairs.flip(-2)

    # concatenate the feature vectors again
    reverse_pairs = reverse_split_pairs.view(pairs.shape)
    return reverse_pairs


class CmpNN(nn.Module):
    r"""A pairwise preference module with weight sharing.

    This module lifts a "core preference module" to a comparative neural
    network with weight sharing. It uses shared weights for the computations
    :math:`U_1(x_1, x_2)` and :math:`U_1(x_2, x_1)`, which should produce a
    more consistent result. The architecture was originally proposed in [1]_.
    That original proposal suggested a single linear layer for the "core
    preference module", but this implementation generalizes the approach to any
    other "zeroth order module".

    The architecture works by evaluating the "core" network on the object
    pairing in both orders (:math:`U_1(x_1, x_2)` and :math:`U_1(x_2, x_1)`)
    and then aggregating the result. The weights of the two evaluations are
    shared, therefore passing the objects in the opposite order will use the
    same weights for the pairwise evaluations and then aggregate those
    evaluations with swapped weights.

    According to the original architecture this network should return two
    outputs: One that indicates the "evidence" that the first input is
    preferable to the second one, and one that indicates the opposite. The two
    outputs would be swapped when the inputs are swapped. For ease of
    integration with other utility modules, we have decided to only return the
    first output. The other output can be accessed by swapping the inputs. That
    is a bit less efficient for inference, but makes it easier to use this as
    an exchangeable component.

    Parameters
    ----------
    pairwise_preference_core: instantiated pytorch module
        This module is used for the pairwise evaluations. The original paper
        [1]_ suggests to use a simple linear layer. This module should expect
        an input of shape :math:`(N, 2 \cdot F)` where :math:`F` is the size of
        a feature vector, and produce an output of shape :math:`(N, C)` where
        :math:`C` is the core output size.
    core_encoding_size: int
        The size of an object encoding that is returned by the pairwise
        preference core.

    References
    ----------
    .. [1] Rigutini, L., Papini, T., Maggini, M., & Scarselli, F. (2011).
    SortNet: Learning to rank by a neural preference function. IEEE
    transactions on neural networks, 22(9), 1368-1380.
    """

    def __init__(self, pairwise_preference_core, core_encoding_size):
        super().__init__()
        # The core of the network. Just one layer in the original paper. See
        # Figure 1 of [1]. Should expect input size 2*n_features and have
        # output size n_hidden.
        self.pairwise_preference_core = pairwise_preference_core
        # Predict the outputs N_<, N_> (depending on the order in which the inputs are passed)
        self.final_layer = nn.Linear(core_encoding_size * 2, 1)
        # self.pairwise_comparison_module = pairwise_comparison_module

    def forward(self, pairs):
        reverse_pairs = _flip_pairs(pairs)
        comparisons = self.pairwise_preference_core(pairs)
        # Both hidden outputs are aggregated in final layer
        reverse_comparisons = self.pairwise_preference_core(reverse_pairs)
        # "x preferable to y", denoted N_> in the paper
        output_one = self.final_layer(torch.cat([comparisons, reverse_comparisons], -1))
        # "y preferable to x", denoted N_< in the paper

        # output_two is equal to output_one of the reversed input (by design).
        # We could use this to build up the pairwise comparison matrix. That
        # would be more efficient, but also a bit more complex. It should only
        # make a difference for inference.

        # output_two = self.final_layer(torch.cat([reverse_comparisons, comparisons], -1))
        # return torch.cat([output_one, output_two], -1)

        # Note that if the inputs are switched then "comparisons" and "reverse
        # comparisons" are switched and therefore also output_one and
        # output_two are switched. That ensures consistency (i.e. that the
        # preference is independent of the order in which the inputs where
        # given).
        return output_one
