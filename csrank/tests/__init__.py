import tensorflow as tf

import pytest


@pytest.fixture(scope='function', autouse=True)
def eager(request):
    tf.compat.v1.enable_eager_execution()
