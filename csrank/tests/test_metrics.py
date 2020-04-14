import itertools

from keras import backend as K
import numpy as np
from numpy.testing import assert_almost_equal
import pytest
from pytest import approx

from csrank.metrics import err
from csrank.metrics import kendalls_tau_for_scores
from csrank.metrics import make_ndcg_at_k_loss
from csrank.metrics import spearman_correlation_for_scores
from csrank.metrics import zero_one_accuracy
from csrank.metrics import zero_one_accuracy_for_scores
from csrank.metrics import zero_one_rank_loss
from csrank.metrics import zero_one_rank_loss_for_scores
from csrank.metrics import zero_one_rank_loss_for_scores_ties
from csrank.metrics_np import err_np
from csrank.metrics_np import kendalls_tau_for_scores_np
from csrank.metrics_np import spearman_correlation_for_scores_np
from csrank.metrics_np import spearman_correlation_for_scores_scipy
from csrank.metrics_np import zero_one_accuracy_for_scores_np
from csrank.metrics_np import zero_one_rank_loss_for_scores_np
from csrank.metrics_np import zero_one_rank_loss_for_scores_ties_np
from csrank.numpy_util import ranking_ordering_conversion


@pytest.fixture(scope="module", params=[(False), (True)], ids=["NoTies", "Ties"])
def problem_for_pred(request):
    ties = request.param
    y_true = np.arange(5)[None, :]
    # We test the error by swapping one adjacent pair:
    if ties:
        y_pred = np.array([[0, 2, 1, 2, 3]])
    else:
        y_pred = np.array([[0, 2, 1, 3, 4]])
    return y_true, y_pred, ties


@pytest.fixture(scope="module", params=[(False), (True)], ids=["NoTies", "Ties"])
def problem_for_scores(request):
    ties = request.param
    y_true = np.arange(5)[None, :]
    # We test the error by swapping one adjacent pair:
    if ties:
        y_scores = np.array([[1.0, 0.8, 0.9, 0.8, 0.6]])
    else:
        y_scores = np.array([[1.0, 0.8, 0.9, 0.7, 0.6]])
    return y_true, y_scores, ties


def test_zero_one_rank_loss(problem_for_pred):
    y_true, y_pred, ties = problem_for_pred
    score = zero_one_rank_loss(y_true, y_pred)
    real_score = K.eval(score)
    if ties:
        assert_almost_equal(actual=real_score, desired=np.array([0.15]))
    else:
        assert_almost_equal(actual=real_score, desired=np.array([0.1]))


def test_zero_one_rank_loss_for_scores(problem_for_scores):
    y_true, y_scores, ties = problem_for_scores
    score = zero_one_rank_loss_for_scores(y_true, y_scores)
    real_score = K.eval(score)
    if ties:
        assert_almost_equal(actual=real_score, desired=np.array([0.15]))
    else:
        assert_almost_equal(actual=real_score, desired=np.array([0.1]))

    score = zero_one_rank_loss_for_scores_ties(y_true, y_scores)
    real_score = K.eval(score)
    if ties:
        assert_almost_equal(actual=real_score, desired=np.array([0.15]))
    else:
        assert_almost_equal(actual=real_score, desired=np.array([0.1]))

    y_true, y_scores, ties = problem_for_scores
    real_score = zero_one_rank_loss_for_scores_np(y_true, y_scores)
    if ties:
        assert_almost_equal(actual=real_score, desired=np.array([0.15]))
    else:
        assert_almost_equal(actual=real_score, desired=np.array([0.1]))

    real_score = zero_one_rank_loss_for_scores_ties_np(y_true, y_scores)
    if ties:
        assert_almost_equal(actual=real_score, desired=np.array([0.15]))
    else:
        assert_almost_equal(actual=real_score, desired=np.array([0.1]))


def test_zero_one_accuracy(problem_for_pred):
    y_true, y_pred, ties = problem_for_pred

    score = zero_one_accuracy(y_true, y_pred)
    real_score = K.eval(score)
    assert_almost_equal(actual=real_score, desired=np.array([0.0]))

    y_true, y_pred, ties = problem_for_pred

    real_score = zero_one_accuracy_for_scores_np(y_true, y_pred)
    assert_almost_equal(actual=real_score, desired=np.array([0.0]))


def test_ndcg(problem_for_pred):
    # ties don't matter here because it doesn't change the two highest predictions
    y_true, y_pred, _ties = problem_for_pred
    # We have:
    # y_true = [0, 1, 2, 3, 4]
    # y_pred = [0, 2, 1, 2, 3]

    # Inverted (with max_rank = 4) that is
    # y_true_inv = [4, 3, 2, 1, 0]
    # y_pred_inv = [4, 2, 3, 2, 1]

    # And normalized to [0, 1] this gives us the relevance:
    # rel_true = [1, 3/4, 1/2, 1/4, 0]
    # rel_pred = [1, 1/2, 3/4, 1/2, 1/4]

    # With this we can first compute the ideal dcg, considering only the first
    # k=2 elements (all logs are base 2, equality is approximate):
    idcg = (2 ** 1 - 1) / np.log2(2) + (2 ** (3 / 4) - 1) / np.log2(3)  # = 1.43

    # And the dcg of the predictions at the same positions as the elements we
    # considered for the idcg (i.e. the "true" best elements):
    dcg = (2 ** 1 - 1) / np.log2(2) + (2 ** (1 / 2) - 1) / np.log2(3)  # = 1.26

    # Now the gain is:
    expected_gain = dcg / idcg  # = 0.882

    ndcg = make_ndcg_at_k_loss(k=2)
    real_gain = K.eval(ndcg(y_true, y_pred))

    assert_almost_equal(actual=real_gain, desired=expected_gain, decimal=5)


def test_kendalls_tau_for_scores(problem_for_scores):
    y_true, y_pred, ties = problem_for_scores

    score = kendalls_tau_for_scores(y_true, y_pred)
    real_score = K.eval(score)
    if ties:
        assert_almost_equal(actual=real_score, desired=np.array([0.7]))
    else:
        assert_almost_equal(actual=real_score, desired=np.array([0.8]))

    real_score = kendalls_tau_for_scores_np(y_true, y_pred)
    if ties:
        assert_almost_equal(actual=real_score, desired=np.array([0.7]))
    else:
        assert_almost_equal(actual=real_score, desired=np.array([0.8]))


def test_spearman_for_scores(problem_for_scores):
    y_true_tensor, y_scores_tensor, ties = problem_for_scores

    score = spearman_correlation_for_scores(y_true_tensor, y_scores_tensor)
    real_score = K.eval(score)
    if ties:
        # We do not handle ties for now
        assert True
    else:
        assert_almost_equal(actual=real_score, desired=np.array([0.9]))

    y_true, y_scores, ties = problem_for_scores

    real_score = spearman_correlation_for_scores_scipy(y_true, y_scores)
    if ties:
        assert_almost_equal(actual=real_score, desired=np.array([0.8207827]))
    else:
        assert_almost_equal(actual=real_score, desired=np.array([0.9]))

    real_score = spearman_correlation_for_scores_np(y_true, y_scores)
    if ties:
        assert True
    else:
        assert_almost_equal(actual=real_score, desired=np.array([0.9]))


def test_zero_one_accuracy_for_scores(problem_for_scores):
    y_true_tensor, y_scores_tensor, ties = problem_for_scores

    score = zero_one_accuracy_for_scores(y_true_tensor, y_scores_tensor)
    real_score = K.eval(score)
    if ties:
        assert_almost_equal(actual=real_score, desired=np.array([0.0]))
    else:
        assert_almost_equal(actual=real_score, desired=np.array([0.0]))


def test_err_perfect_first_trumps_many_good():
    """Tests that a perfect document at rank 1 trumps later rankings.

    The authors of [1] list this as a motivating example. A ranking that
    puts a "perfect" document at rank 1 (i.e. one that is almost certain
    to satisfy the user's needs) should trump one that puts a "good" one
    at rank 1, regardless of the documents at later ranks. The reasoning
    is that later ranks won't need to be examined when the first is
    already sufficient.

    References
    ----------
        [1] Chapelle, Olivier, et al. "Expected reciprocal rank for graded
        relevance." Proceedings of the 18th ACM conference on Information and
        knowledge management. ACM, 2009. http://olivier.chapelle.cc/pub/err.pdf
    """
    y_true = ranking_ordering_conversion([range(20)])

    # gets the "perfect" one right, everything else wrong
    perfect_first = ranking_ordering_conversion(
        [[0, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]]
    )

    # does pretty good for most, but ranks the "perfect" one wrong
    all_good = ranking_ordering_conversion(
        [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 0]]
    )

    assert K.eval(err(y_true, perfect_first)) > K.eval(err(y_true, all_good))


def test_err_against_manually_verified_example():
    """Compares the implementation against a manual calculation."""
    y_true = ranking_ordering_conversion([[1, 2, 0]])
    y_pred = ranking_ordering_conversion([[2, 1, 0]])
    # The resulting probabilities that each document satisfies the
    # user's need:
    # [2**1-1, 2**2-1, 2**0 - 1] / 2**2 = [1/4, 3/4, 0]
    # Multiplied by the respective rank utilities (1/(r+1)):
    # [(1/4)/3, (3/4)/2, 0/1] = [1/12, 3/8, 0]
    # The resulting ERR:

    # We ranked object 2 first, which has a true rank of 1 and therefore
    # (with the relevance gain probability mapping) a probability of
    # (2**(2-1)-1) / 2**2 = 1/4
    # of matching the user's need. It is at rank 0, which has utility
    # 1/(0+1) = 1.

    # Object 1 is next. True rank of 0, probability
    # (2**(2-0)-1) / 2**2 = 3/4
    # and utility
    # 1/(1+1) = 1/2.

    # Object 0 last. True rank of 2, probability
    # (2**(2-2)-1) / 2**2 = 0
    # and utility
    # 1/(2+1) = 1/3.

    # The resulting expected utility:
    # 1/4 * 1 + (1 - 1/4) * 3/4 * 1/2 + (1 - 1/4) * (1 - 3/4) * 0 * 1/3
    # = 17/32
    # Approx because comparing floats is inherently error-prone.
    assert K.eval(err(y_true, y_pred)) == approx(17 / 32)


def test_err_implementations_equivalent():
    """Spot-checks equivalence of plain python and tf implementations"""
    # A simple grading where each grade occurs once. We want to check
    # for equivalence at every permutation of this grading.
    elems = np.array([4, 3, 2, 1, 0])
    y_true = np.reshape(elems, (1, -1))
    # Spot check some permutations (5! / 20 = 6 checks are performed)
    for perm in list(itertools.permutations(elems))[::20]:
        perm = np.reshape(perm, (1, -1))
        assert K.eval(err(y_true, perm)) == approx(err_np(y_true, perm))
