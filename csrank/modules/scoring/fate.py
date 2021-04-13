"""An implementation of the scoring module for FATE estimators."""

import functools

import torch
import torch.nn as nn

from csrank.modules.instance_reduction import DeepSet
from csrank.modules.object_mapping import DenseNeuralNetwork


class FATEScoring(nn.Module):
    r"""Map instances to scores with the FATE approach.

    Let's show the FATE approach on an example. To simplify things, we'll use a
    simply identity-embedding. The FATE module will then aggregate the context
    by simply taking the average of the objects (feature-wise). To further
    simplify things the actual pairwise utility is just computed by the sum of
    all features of the object and the context.

    >>> import torch.nn as nn
    >>> from csrank.modules.object_mapping import DeterministicSumming
    >>> scoring = FATEScoring(
    ...     n_features=2,
    ...     pairwise_utility_module=DeterministicSumming,
    ...     embedding_module=nn.Identity,
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

    Let's focus on the first instance in this example. The aggregated identity
    embedding is

    >>> embedding_1 = (object_a[0] + object_b[0] + object_c[0]) / 3
    >>> embedding_2 = (object_a[1] + object_b[1] + object_c[1]) / 3
    >>> (embedding_1, embedding_2)
    (1.5, 1.8)

    for the first and second feature respectively. So the utility of object_a
    within the context (defined by the mock sum utility) should be

    >>> embedding_1 + embedding_2 + object_a[0] + object_a[1]
    4.6

    Let's verify this:

    >>> scoring(instances)
    tensor([[ 4.6000,  6.6000,  8.6000],
            [16.2000, 18.2000, 20.2000]])

    As you can see, the scoring comes to the same result for the first object
    of the first instance.

    Parameters
    ----------
    n_features: int
        The number of features each object has.
    embedding_size: int
        The size of the embeddings that should be generated. Defaults to
        ``n_features`` if not specified.
    pairwise_utility_module: pytorch module with one integer parameter
        The module that should be used for pairwise utility estimations. Uses a
        simple linear mapping not specified. You likely want to replace this
        with something more expressive such as a ``DenseNeuralNetwork``. This
        should take the size of the input values as its only parameter. You can
        use ``functools.partial`` if necessary. This corresponds to
        :math:`U` in Figure 2 of [1]_.
    embedding_module: pytorch module with one integer parameter
        The module that should be used for the object embeddings. Its
        constructor should take two parameters: The size of the input and the
        size of the output. This corresponds to :math:`\Phi` in Figure 2 of
        [1]_. The default is a ``DenseNeuralNetwork`` with 5 hidden layers and
        64 units per hidden layer.

    References
    ----------
    .. [1] Pfannschmidt, K., Gupta, P., & Hüllermeier, E. (2019). Learning
    choice functions: Concepts and architectures. arXiv preprint
    arXiv:1901.10860.
    """

    def __init__(
        self,
        n_features,
        embedding_size=None,
        pairwise_utility_module=None,
        embedding_module=None,
    ):
        super().__init__()
        if embedding_size is None:
            embedding_size = n_features
        if pairwise_utility_module is None:
            pairwise_utility_module = functools.partial(
                nn.Linear,
                out_features=1,
            )
        if embedding_module is None:
            embedding_module = functools.partial(
                DenseNeuralNetwork, hidden_layers=5, units_per_hidden=64
            )

        self.embedding = DeepSet(
            n_features,
            embedding_size,
            embedding_module=embedding_module,
        )
        self.pairwise_utility_module = pairwise_utility_module(
            n_features + embedding_size
        )

    def forward(self, instances, **kwargs):
        n_objects = instances.size(1)
        contexts = self.embedding(instances)
        # Repeat each context for each object within the instance; This is then
        # a flat list of contexts. Then reshape to have a list of contexts per
        # instance.
        context_per_object = contexts.repeat_interleave(n_objects, dim=0).reshape_as(
            instances
        )
        pairs = torch.stack((instances, context_per_object), dim=-1)
        utilities = self.pairwise_utility_module(pairs.flatten(start_dim=-2)).squeeze()
        return utilities
