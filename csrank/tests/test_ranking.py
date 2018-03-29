from abc import ABCMeta

import numpy as np
import pytest
from keras.layers import Input
from keras.models import Model
from keras.optimizers import SGD
from keras.regularizers import l2

from ..fate_ranking import FATERankingCore, FATEObjectRanker


def test_construction_core():
    n_objects = 3
    n_features = 2

    # Create mock class:
    class MockClass(FATERankingCore, metaclass=ABCMeta):
        def set_tunable_parameters(self, point):
            super().set_tunable_parameters(point)

        def predict(self, *args, **kwargs):
            pass

        def fit(self, *args, **kwargs):
            pass

    grc = MockClass(n_objects=n_objects, n_features=n_features)
    grc._construct_layers(activation=grc.activation,
                          kernel_initializer=grc.kernel_initializer,
                          kernel_regularizer=grc.kernel_regularizer)
    input_layer = Input(shape=(n_objects, n_features))
    scores = grc.join_input_layers(input_layer,
                                   None,
                                   n_layers=0,
                                   n_objects=n_objects)

    model = Model(inputs=input_layer, outputs=scores)
    model.compile(loss='mse', optimizer=grc.optimizer)
    X = np.random.randn(100, n_objects, n_features)
    y = X.sum(axis=2)
    model.fit(x=X, y=y, verbose=0)


@pytest.fixture(scope="module")
def trivial_ranking_problem():
    rand = np.random.RandomState(123)
    x = rand.randn(100, 5, 1)
    y_true = x.argsort(axis=1).argsort(axis=1).squeeze(axis=-1)
    return x, y_true


def test_fate_object_ranker_fixed(trivial_ranking_problem):
    x, y = trivial_ranking_problem
    fate = FATEObjectRanker(n_object_features=1,
                            n_hidden_joint_layers=1,
                            n_hidden_set_layers=1,
                            n_hidden_joint_units=5,
                            n_hidden_set_units=5,
                            kernel_regularizer=l2(1e-4),
                            optimizer=SGD(lr=1e-3, momentum=0.9, nesterov=True))
    fate.fit(x, y, epochs=50, validation_split=0, verbose=False)
    pred = fate.predict(x)
    assert np.all(pred == y)


def test_fate_object_ranker_fixed_generator():
    def trivial_ranking_problem_generator():
        while True:
            rand = np.random.RandomState(123)
            x = rand.randn(10, 5, 1)
            y_true = x.argsort(axis=1).argsort(axis=1).squeeze(axis=-1)
            yield x, y_true

    fate = FATEObjectRanker(n_object_features=1,
                            n_hidden_joint_layers=1,
                            n_hidden_set_layers=1,
                            n_hidden_joint_units=5,
                            n_hidden_set_units=5,
                            kernel_regularizer=l2(1e-4),
                            optimizer=SGD(lr=1e-3, momentum=0.9, nesterov=True))
    fate.fit_generator(generator=trivial_ranking_problem_generator(),
                       epochs=1, validation_split=0, verbose=False,
                       steps_per_epoch=10)
