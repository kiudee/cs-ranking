"""An implementation of the scoring module for FATE estimators."""

import functools

import torch.nn as nn

from csrank.modules.collection import PairwiseEvaluation
from csrank.modules.instance_reduction import MeanAggregatedUtility
from csrank.modules.object_mapping import CmpNN
from csrank.modules.object_mapping import DenseNeuralNetwork


class CmpNetScoring(nn.Module):
    r"""Map instances to scores with the CmpNet approach.

    This approach was introduced in [1]_. It is very similar to the FETA (first
    evaluate, then aggregate) approach. It uses a ``CmpNet`` for evaluation.
    The network returns pairwise "evidence" that the first object is preferable
    to the second one. That is conceptually the inverse of a pairwise utility
    ("how much does the presence of one object support the utility of
    another?"), but we can ignore that in the implementation since the two are
    equivalently learnable by a neural network. The ``CmpNet`` does not return
    preferences directly but rather "evidence" for preference. That means that
    the preferences are not normalized, which again does not matter for the
    relative scores.

    The evidences of one object against all others are then aggregated by a
    mean reduction, resulting in something similar to a Borda score for each
    object. These scores can be used for ranking and choice tasks. Not that
    this differs from the sorting approach taken in [1]_.

    The main practical difference to the FETA approach is the weight-sharing in
    the evaluation. See the documentation of ``CmpNet`` for details.

    Parameters
    ----------
    n_features: int
        The number of features each object has.
    pairwise_preference_core: instantiated pytorch module
        This module is used for the pairwise evaluations. The original paper
        [1]_ suggests to use a simple linear layer. This module should expect
        an input of shape :math:`(N, 2 \cdot F)` where :math:`F` is the size of
        a feature vector, and produce an output of shape :math:`(N, C)` where
        :math:`C` is the core output size. The output will be further processed
        by the ``CmpNet``.
    core_encoding_size: int
        The size of the embedding that the pairwise preference core should
        produce before the results are aggregated by the final layer. See the
        documentation of ``CmpNet`` for a more detailed understanding.

    References
    ----------
    .. [1] Rigutini, L., Papini, T., Maggini, M., & Scarselli, F. (2011).
    SortNet: Learning to rank by a neural preference function. IEEE
    transactions on neural networks, 22(9), 1368-1380.
    """

    def __init__(
        self,
        n_features,
        pairwise_preference_core=functools.partial(
            DenseNeuralNetwork, hidden_layers=1, units_per_hidden=8
        ),
        core_encoding_size=8,
    ):
        super().__init__()
        self.mean_aggregated_evidence = MeanAggregatedUtility(
            PairwiseEvaluation(
                CmpNN(
                    pairwise_preference_core(2 * n_features, core_encoding_size),
                    core_encoding_size=core_encoding_size,
                )
            ),
            exclude_self_comparison=True,
        )

    def forward(self, instances, **kwargs):
        return self.mean_aggregated_evidence(instances)
