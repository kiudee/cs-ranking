import numpy as np
import tensorflow as tf
from keras import backend as K

from ..util import tensorify


def test_tensorify():
    a = np.array([1., 2.])
    out = tensorify(a)
    assert isinstance(out, tf.Tensor)

    b = K.zeros((5, 3))
    out = tensorify(b)
    assert b == out
