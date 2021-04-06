"""Modules that reduce instances to some kind of feature representation.

The modules listed here should take a 2 or higher dimensional input and reduce
the second-to-last dimension. They can take interaction of "elements" (feature
vectors) into account.

Inputs of shape :math:`(N, *, O, H_i)` are transformed to outputs of shape
:math:`(N, *, H_o)`. In this case :math:`N` is the batch size, :math:`*`
denotes arbitrary additional dimension (which are preserved), :math:`O` refers
to the number of objects per instance (the reduced dimension) and :math:`H_i`
and :math:`H_o` refer to the number of input and output features respectively.
"""

import torch
import torch.nn as nn


class DeepSet(nn.Module):
    """Aggregate object-level embeddings with a mean reduction.

    This module evaluates each object individually (using a object level
    embedding) and then aggregates the embeddings with a mean reduction.

    Parameters
    ----------
    n_features : int
        The number of features per object.

    embedding_size : int
        The target embedding size.

    embedding_module : torch module
        An uninitialized torch module that expects two parameters: the input
        and the output size. It should then act similar to ``nn.Linear``, i.e.
        transform only the last dimension of the input. Defaults to a simple
        linear module.
    """

    def __init__(
        self,
        n_features: int,
        embedding_size: int,
        embedding_module: nn.Module = nn.Linear,
    ):
        super().__init__()
        self.embedding_module = embedding_module(n_features, embedding_size)

    def forward(self, instances):
        """Forward inputs through the network.

        Parameters
        ----------
        instances : tensor
            The input tensor of shape (N, *, O, F), where F is the number of
            features and O is the number of objects.

        Returns
        -------
        tensor
            A tensor of shape (N, *, E), where E ist the embedding size.
        """
        embedded_objects = self.embedding_module(instances)
        return torch.mean(embedded_objects, dim=1)


class MeanAggregatedUtility(nn.Module):
    """Map instances to individual utilities by a first order utility function and aggregation.

    This is an approximation of a generalized utility function as described in
    section 3.1 of [1]_. This is the "Then Aggregate" part of FETA. Only
    aggregation of first-order utility functions is supported.

    Parameters
    ----------
    first_order_utility : torch module (instantiated)
        A function that maps instances to a list of utility matrices. The
        resulting matrix will be interpreted as the utility each object has,
        given the context of the other object.

        The function should expect input in the shape of [N, n_features] and
        return output in the shape of [N]. That is, it should map a list of
        concatenated object features to the pairwise utilities of these
        objects.
    exclude_self_comparison : bool
        Whether or not to exclude the evaluations of the objects with
        themselves (i.e. the diagonal values) when aggregating.

    Returns
    -------
    A list of instances with the aggregated object utilities.

    References
    ----------
    .. [1] Pfannschmidt, K., Gupta, P., & Hüllermeier, E. (2019). Learning
    choice functions: Concepts and architectures. arXiv preprint
    arXiv:1901.10860.
    """

    def __init__(self, first_order_utility, exclude_self_comparison):
        super().__init__()
        self.first_order_utility = first_order_utility
        self.exclude_self_comparison = exclude_self_comparison

    def forward(self, instances, **kwargs):
        """Aggregate zeroth and first order utility, returning a 1d tensor of evaluations.

        Parameters
        ----------
        instances : tensor
            The input tensor of shape (N, *, O, F), where F is the number of
            features and O is the number of objects.

        Returns
        -------
        tensor
            The aggregated utility of each object: A tensor of shape (N, *, O).
        """
        utility_matrix = self.first_order_utility(instances)
        n_objects = utility_matrix.shape[-1]
        n_instances = utility_matrix.shape[0]
        if self.exclude_self_comparison:
            # One diagonal unit matrix for each instance
            diagonal_mask = (
                torch.eye(n_objects, dtype=torch.uint8)
                .view(1, n_objects, n_objects)
                .repeat(n_instances, 1, 1)
            )
            # Fill the diagonal of each instance with 0 (to ignore it while
            # summing).
            utility_matrix[diagonal_mask] = 0
            # The diagonal values do not count when computing the mean.
            n_objects -= 1
        # Compute the mean utility of each object, possibly ignoring the
        # diagonal values.
        context_utility = utility_matrix.sum(dim=-1) / n_objects
        return context_utility
