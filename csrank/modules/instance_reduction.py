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
