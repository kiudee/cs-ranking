"""Loss functions for discrete choice problems."""

import torch


class CategoricalHingeLossMax:
    """Compute the Categorical Hinge Loss.

    This is the "max" aggregated version of CHL, described on page 14/15 of
    [1]_.

    Parameters
    ----------
    scores: 2d tensor
        The predicted scores for each object of each instance.

    true_choice: 2d tensor
        The true choice mask for each instance.

    Returns
    -------
    torch.float
        The total loss, summed over all instances.

    References
    ----------
    .. [1] Pfannschmidt, K., Gupta, P., & Hüllermeier, E. (2019). Learning
    choice functions: Concepts and architectures. arXiv preprint
    arXiv:1901.10860.
    """

    # The argument order is chosen to be compatible with skorch.
    def __call__(self, scores, true_choice):
        """Compute the loss of a scoring in the context of a choice.

        >>> objects = ["a", "b", "c"]
        >>> true_choice_1 = [0, 1, 0]
        >>> scores_1 = [2, 1, 0.5] # non-chosen object "a" has higher score than chosen object "b"
        >>> chl = CategoricalHingeLossMax()
        >>> chl(torch.tensor([scores_1]), torch.tensor([true_choice_1]))
        tensor(2.)

        >>> true_choice_2 = [1, 0, 0]
        >>> scores_2 = [0, 1, 1.5]

        >>> chl = CategoricalHingeLossMax()
        >>> chl(torch.tensor([scores_1, scores_2]), torch.tensor([true_choice_1, true_choice_2]))
        tensor(4.5000)
        """
        # not quite, but dealing with true infintiy is hairy and there should
        # be no practical difference
        infty = 2 ** 32

        # Mask out the chosen scores from the max with a value of -infinity.
        (max_score_not_chosen, _indices) = torch.max(
            scores - true_choice * infty, dim=1
        )
        # Mask out the not-chosen scores from the min with a value of +infinity.
        (min_score_chosen, _indices) = torch.min(
            scores + (1 - true_choice) * infty, dim=1
        )

        hinge = torch.clamp(1 + max_score_not_chosen - min_score_chosen, min=0)
        return hinge.sum()
