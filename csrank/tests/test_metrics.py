import pytest
import numpy as np
from keras import backend as K
from numpy.testing import assert_almost_equal

from csrank.metrics import zero_one_rank_loss, zero_one_rank_loss_for_scores, \
    zero_one_accuracy, make_ndcg_at_k_loss


@pytest.fixture(scope="module",
                params=[(True, False), (False, True)],
                ids=['NoTies', 'Ties'])
def problem_for_pred(request):
    is_numpy, ties = request.param
    y_true = np.arange(5)[None, :]
    # We test the error by swapping one adjacent pair:
    if ties:
        y_pred = np.array([[0, 2, 1, 2, 3]])
    else:
        y_pred = np.array([[0, 2, 1, 3, 4]])
    if is_numpy:
        return y_true, y_pred, ties
    y_true_tensor = K.constant(y_true)
    y_pred_tensor = K.constant(y_pred)
    return y_true_tensor, y_pred_tensor, ties


@pytest.fixture(scope="module",
                params=[(True, False), (False, True)],
                ids=['NoTies', 'Ties'])
def problem_for_scores(request):
    is_numpy, ties = request.param
    y_true = np.arange(5)[None, :]
    # We test the error by swapping one adjacent pair:
    if ties:
        y_scores = np.array([[1., 0.8, 0.9, 0.8, 0.6]])
    else:
        y_scores = np.array([[1., 0.8, 0.9, 0.7, 0.6]])
    if is_numpy:
        return y_true, y_scores, ties
    y_true_tensor = K.constant(y_true)
    y_scores_tensor = K.constant(y_scores)
    return y_true_tensor, y_scores_tensor, ties


def test_zero_one_rank_loss(problem_for_pred):
    y_true_tensor, y_pred_tensor, ties = problem_for_pred
    score = zero_one_rank_loss(y_true_tensor, y_pred_tensor)
    real_score = K.eval(score)
    if ties:
        assert_almost_equal(actual=real_score, desired=np.array([0.15]))
    else:
        assert_almost_equal(actual=real_score, desired=np.array([0.1]))


def test_zero_one_rank_loss_for_scores(problem_for_scores):
    y_true_tensor, y_scores_tensor, ties = problem_for_scores

    score = zero_one_rank_loss_for_scores(y_true_tensor, y_scores_tensor)
    real_score = K.eval(score)
    if ties:
        assert_almost_equal(actual=real_score, desired=np.array([0.15]))
    else:
        assert_almost_equal(actual=real_score, desired=np.array([0.1]))


def test_zero_one_accuracy(problem_for_pred):
    y_true_tensor, y_pred_tensor, ties = problem_for_pred

    score = zero_one_accuracy(y_true_tensor, y_pred_tensor)
    real_score = K.eval(score)
    assert_almost_equal(actual=real_score, desired=np.array([0.]))


def test_ndcg(problem_for_pred):
    y_true_tensor, y_pred_tensor, ties = problem_for_pred

    ndcg = make_ndcg_at_k_loss(k=2)
    gain = ndcg(y_true_tensor, y_pred_tensor)
    real_gain = K.eval(gain)

    expected_dcg = 15. + 3. / np.log2(3.)
    expected_idcg = 15. + 7. / np.log2(3.)
    assert_almost_equal(actual=real_gain,
                        desired=np.array([[expected_dcg/expected_idcg]]),
                        decimal=5)
