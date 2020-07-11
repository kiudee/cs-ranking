from keras import backend as K
import numpy as np
import tensorflow as tf

from csrank import SyntheticIterator
from csrank.tensorflow_util import tensorify


def test_tensorify():
    a = np.array([1.0, 2.0])
    out = tensorify(a)
    assert isinstance(out, tf.Tensor)

    b = K.zeros((5, 3))
    out = tensorify(b)
    assert b == out


def test_synthetic_iterator():
    def func(a, b):
        return (b, a)

    it = SyntheticIterator(dataset_function=func, a=41, b=2)
    for i, (x, y) in enumerate(it):
        if i == 1:
            break
        assert x == 2
        assert y == 41
