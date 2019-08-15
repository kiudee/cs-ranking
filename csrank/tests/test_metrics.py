import numpy as np
import pytest
from keras import backend as K
from numpy.testing import assert_almost_equal

from csrank.metrics import zero_one_rank_loss, zero_one_rank_loss_for_scores, zero_one_rank_loss_for_scores_ties, \
    zero_one_accuracy, make_ndcg_at_k_loss, kendalls_tau_for_scores, \
    spearman_correlation_for_scores, zero_one_accuracy_for_scores
from csrank.metrics_np import zero_one_rank_loss_for_scores_np, zero_one_rank_loss_for_scores_ties_np, \
    spearman_correlation_for_scores_np, spearman_correlation_for_scores_scipy, kendalls_tau_for_scores_np, \
    zero_one_accuracy_for_scores_np


@pytest.fixture(scope="module",
                params=[(False), (True)],
                ids=['NoTies', 'Ties'])
def problem_for_pred(request):
    ties = request.param
    y_true = np.arange(5)[None, :]
    # We test the error by swapping one adjacent pair:
    if ties:
        y_pred = np.array([[0, 2, 1, 2, 3]])
    else:
        y_pred = np.array([[0, 2, 1, 3, 4]])
    return y_true, y_pred, ties


@pytest.fixture(scope="module",
                params=[(False), (True)],
                ids=['NoTies', 'Ties'])
def problem_for_scores(request):
    ties = request.param
    y_true = np.arange(5)[None, :]
    # We test the error by swapping one adjacent pair:
    if ties:
        y_scores = np.array([[1., 0.8, 0.9, 0.8, 0.6]])
    else:
        y_scores = np.array([[1., 0.8, 0.9, 0.7, 0.6]])
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
    assert_almost_equal(actual=real_score, desired=np.array([0.]))

    y_true, y_pred, ties = problem_for_pred

    real_score = zero_one_accuracy_for_scores_np(y_true, y_pred)
    assert_almost_equal(actual=real_score, desired=np.array([0.]))


@pytest.mark.skip("Current code had numeric problems and needs to be rewritten")
def test_ndcg(problem_for_pred):
    y_true, y_pred, ties = problem_for_pred

    ndcg = make_ndcg_at_k_loss(k=2)
    gain = ndcg(y_true, y_pred)
    real_gain = K.eval(gain)

    expected_dcg = 15. + 3. / np.log2(3.)
    expected_idcg = 15. + 7. / np.log2(3.)
    assert_almost_equal(actual=real_gain,
                        desired=np.array([[expected_dcg / expected_idcg]]),
                        decimal=5)


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
