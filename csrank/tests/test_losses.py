import numpy as np
from keras import backend as K
from numpy.testing import assert_almost_equal

from csrank.losses import hinged_rank_loss, smooth_rank_loss


def test_hinged_rank_loss():
    y_true = np.arange(5)[None, :]
    y_true_tensor = K.constant(y_true)

    # Predicting all 0, gives an error of 1.0:
    assert_almost_equal(
        actual=K.eval(
            hinged_rank_loss(y_true_tensor,
                             K.constant(np.array([[0., 0., 0., 0., 0.]])))),
        desired=np.array([1.]))

    # Predicting the correct ranking improves, but penalizes by difference of
    # scores:
    assert_almost_equal(
        actual=K.eval(
            hinged_rank_loss(y_true_tensor,
                             K.constant(np.array([[.2, .1, .0, -0.1, -0.2]])))),
        desired=np.array([0.8]))


def test_smooth_rank_loss():
    y_true = np.arange(5)[None, :]
    y_true_tensor = K.constant(y_true)

    # Predicting all 0, gives an error of 1.0:
    assert_almost_equal(
        actual=K.eval(
            smooth_rank_loss(y_true_tensor,
                             K.constant(np.array([[0., 0., 0., 0., 0.]])))),
        desired=np.array([1.]))

    # Predicting the correct ranking improves, but penalizes by difference of
    # scores:
    assert_almost_equal(
        actual=K.eval(
            smooth_rank_loss(y_true_tensor,
                             K.constant(np.array([[.2, .1, .0, -0.1, -0.2]])))),
        desired=np.array([0.822749841877]))
