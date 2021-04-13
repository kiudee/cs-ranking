"""Loss functions for ranking choice problems."""

import torch


class HingedRankLoss:
    """Compute the pairwise loss between two lists of rankings.

    Assumes the true ranking is represented by a permutation array (list of
    indices). The scores are floats, which should be high for low ranking
    indices.

    Their relative order of the scores (i.e. their reversed argsort) and their
    difference is important for the resulting loss. Even scores that result in
    a correct ranking will be penalized if the scores of two elements differ by
    less than 1. For example if we have objects

    >>> objects = ["a", "b", "c"]

    then the true ranking (permutation array)

    >>> ranking = [0, 2, 1]

    and the scores

    >>> scores = [0, -10, -5]

    would equivalently specify the ranking ["a, "c", "b"] and consequently

    >>> hinged_rank_loss = HingedRankLoss()
    >>> hinged_rank_loss(torch.tensor([scores]), torch.tensor([ranking]))
    tensor(0.)

    A rescaled scoring with an insufficient gap would lead to a non-zero loss:

    >>> scores = [0, -1, -0.5]
    >>> hinged_rank_loss(torch.tensor([scores]), torch.tensor([ranking]))
    tensor(0.3333)

    The ranking

    >>> ranking = [0, 2, 1]

    is not matched by the scoring

    >>> scores = [-2, 0, -1]

    and thus

    >>> hinged_rank_loss(torch.tensor([scores]), torch.tensor([ranking]))
    tensor(2.3333)

    Parameters
    ----------
    comparison_scores: 2d tensor
        The predicted scores for each object of each instance.

    true_rankings: 2d tensor
        The true rankings, represented as a permutation. The first element of a
        permutation contains the index to which the first element should be
        moved.

    Returns
    -------
    torch.float
        The total loss, summed over all instances.
    """

    # The argument order is chosen to be compatible with skorch.
    def __call__(self, comparison_scores, true_rankings):
        # 2d matrix which is 1 if the row-element *should* be ranked higher than the column element
        mask = true_rankings[:, :, None] > true_rankings[:, None]
        # How much higher/lower the elements are actually ranked. First create
        # new dimensions (at the element/instance level), then rely on
        # broadcasting to compute the difference.
        # Negated because higher scores imply lower rankings.
        diff = -(comparison_scores[:, :, None] - comparison_scores[:, None])
        self.diff = diff
        hinge = torch.clamp(mask * (1 - diff), min=0)
        n = torch.sum(mask, axis=(1, 2))
        losses = torch.true_divide(torch.sum(hinge, axis=(1, 2)), n)
        return losses.sum()
