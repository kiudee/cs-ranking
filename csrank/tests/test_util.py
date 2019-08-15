import numpy as np
import pytest
import tensorflow as tf
from keras import backend as K

from csrank import SyntheticIterator
from csrank.tensorflow_util import tensorify
from csrank.tuning import check_learner_class


def test_tensorify():
    a = np.array([1., 2.])
    out = tensorify(a)
    assert isinstance(out, tf.Tensor)

    b = K.zeros((5, 3))
    out = tensorify(b)
    assert b == out


def test_synthetic_iterator():
    def func(a, b):
        return (b, a)

    it = SyntheticIterator(dataset_function=func,
                           a=41, b=2)
    for i, (x, y) in enumerate(it):
        if i == 1:
            break
        assert x == 2
        assert y == 41


def test_check_ranker_class():
    class MockClass(object):
        def __init__(self):
            pass

    ranker = MockClass()
    with pytest.raises(AttributeError):
        check_learner_class(ranker)
