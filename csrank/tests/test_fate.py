from abc import ABCMeta

import numpy as np
from keras import Input, Model
from keras.regularizers import l2

from csrank import FATENetworkCore, FATEObjectRanker
from csrank.tests.test_ranking import optimizer


def test_construction_core():
    n_objects = 3
    n_features = 2

    # Create mock class:

    class MockClass(FATENetworkCore, metaclass=ABCMeta):

        def set_tunable_parameters(self, **point):
            super().set_tunable_parameters(**point)

        def predict_scores(self, X, **kwargs):
            pass

        def _predict_scores_fixed(self, X, **kwargs):
            pass

        def predict(self, *args, **kwargs):
            pass

        def fit(self, *args, **kwargs):
            pass

    grc = MockClass(n_objects=n_objects, n_features=n_features)
    grc._construct_layers(
        activation=grc.activation,
        kernel_initializer=grc.kernel_initializer,
        kernel_regularizer=grc.kernel_regularizer,
    )
    input_layer = Input(shape=(n_objects, n_features))
    scores = grc.join_input_layers(input_layer, None, n_layers=0, n_objects=n_objects)

    model = Model(inputs=input_layer, outputs=scores)
    model.compile(loss="mse", optimizer=grc.optimizer)
    X = np.random.randn(100, n_objects, n_features)
    y = X.sum(axis=2)
    model.fit(x=X, y=y, verbose=0)
    params = {"n_hidden_joint_units": 2, "n_hidden_joint_layers": 10, "reg_strength": 1e-3, "learning_rate": 1e-1,
              "batch_size": 32}
    grc.set_tunable_parameters(**params)
    assert grc.n_hidden_joint_units == params["n_hidden_joint_units"]
    assert grc.n_hidden_joint_layers == params["n_hidden_joint_layers"]
    assert grc.batch_size == params["batch_size"]
    rtol = 1e-2
    atol = 1e-4
    assert np.isclose(grc.optimizer.get_config()['lr'], params["learning_rate"], rtol=rtol, atol=atol, equal_nan=False)
    config = grc.kernel_regularizer.get_config()
    val1 = np.isclose(config["l1"], params["reg_strength"], rtol=rtol, atol=atol, equal_nan=False)
    val2 = np.isclose(config["l2"], params["reg_strength"], rtol=rtol, atol=atol, equal_nan=False)
    assert val1 or val2


def test_fate_object_ranker_fixed_generator():
    def trivial_ranking_problem_generator():
        while True:
            rand = np.random.RandomState(123)
            x = rand.randn(10, 5, 1)
            y_true = x.argsort(axis=1).argsort(axis=1).squeeze(axis=-1)
            yield x, y_true

    fate = FATEObjectRanker(
        n_object_features=1,
        n_hidden_joint_layers=1,
        n_hidden_set_layers=1,
        n_hidden_joint_units=5,
        n_hidden_set_units=5,
        kernel_regularizer=l2(1e-4),
        optimizer=optimizer,
    )
    fate.fit_generator(
        generator=trivial_ranking_problem_generator(),
        epochs=1,
        validation_split=0,
        verbose=False,
        steps_per_epoch=10,
    )
