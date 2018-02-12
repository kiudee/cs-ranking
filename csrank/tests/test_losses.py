import numpy as np
from keras import backend as K
from numpy.testing import assert_almost_equal

from csrank.metrics import zero_one_rank_loss, zero_one_rank_loss_for_scores, \
    zero_one_rank_loss_for_scores_ties


def test_zero_one_rank_loss():
    y_true = np.arange(5)[None, :]
    y_true_tensor = K.constant(y_true)
    # We test the error by swapping one adjacent pair:
    y_pred = np.array([[0, 2, 1, 3, 4]])
    y_pred_tensor = K.constant(y_pred)

    score = zero_one_rank_loss(y_true_tensor, y_pred_tensor)
    real_score = K.eval(score)
    assert_almost_equal(actual=real_score, desired=np.array([0.1]))

    # Make sure the loss function also works with NumPy arrays:
    zero_one_rank_loss(y_true, y_pred)


def test_zero_one_rank_loss_for_scores():
    y_true = np.arange(5)[None, :]
    y_true_tensor = K.constant(y_true)
    # We test the error by swapping one adjacent pair:
    y_scores = np.array([[1., 0.8, 0.9, 0.7, 0.6]])
    y_scores_tensor = K.constant(y_scores)

    score = zero_one_rank_loss_for_scores(y_true_tensor, y_scores_tensor)
    real_score = K.eval(score)
    assert_almost_equal(actual=real_score, desired=np.array([0.1]))

    # Make sure the loss function also works with NumPy arrays:
    zero_one_rank_loss_for_scores(y_true, y_scores)


def test_zero_one_rank_loss_for_scores_ties():
    y_true = np.arange(5)[None, :]
    y_true_tensor = K.constant(y_true)
    # We test the error by swapping one adjacent pair:
    y_scores = np.array([[1., 0.8, 0.9, 0.8, 0.6]])
    y_scores_tensor = K.constant(y_scores)

    score = zero_one_rank_loss_for_scores_ties(y_true_tensor, y_scores_tensor)
    real_score = K.eval(score)
    assert_almost_equal(actual=real_score, desired=np.array([0.15]))

    # Make sure the loss function also works with NumPy arrays:
    zero_one_rank_loss_for_scores(y_true, y_scores)
