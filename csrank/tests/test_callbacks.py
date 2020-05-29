import math
import os

from keras import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
import numpy as np
import pytest
import tensorflow as tf

from csrank.callbacks import EarlyStoppingWithWeights
from csrank.callbacks import LRScheduler
from csrank.tests.test_ranking import check_params_tunable

callbacks_dict = {
    "EarlyStopping": (EarlyStoppingWithWeights, {"patience": 5, "min_delta": 5e-2}),
    "LRScheduler": (LRScheduler, {"epochs_drop": 5, "drop": 0.9}),
}


@pytest.fixture(scope="module")
def trivial_classification_problem():
    random_state = np.random.RandomState(123)
    x = random_state.randn(200, 2)
    w = random_state.rand(2)
    y = 1.0 / (1.0 + np.exp(-np.dot(x, w)))
    y_true = np.array(y > 0.5, dtype=np.int64)
    return x, y_true


def create_model():
    lr = 0.015
    model = Sequential()
    model.add(Dense(10, activation="relu"))
    model.add(Dense(5, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(optimizer=SGD(lr=lr), loss="binary_crossentropy")
    return model, lr


@pytest.mark.parametrize("name", list(callbacks_dict.keys()))
def test_callbacks(trivial_classification_problem, name):
    tf.set_random_seed(0)
    os.environ["KERAS_BACKEND"] = "tensorflow"
    np.random.seed(123)
    x, y = trivial_classification_problem
    epochs = 15
    model, init_lr = create_model()
    callback, params = callbacks_dict[name]
    callback = callback(**params)
    callbacks = [callback]
    model.fit(x, y, epochs=epochs, callbacks=callbacks, validation_split=0.1)
    rtol = 1e-2
    atol = 5e-4
    if name == "LRScheduler":
        epochs_drop, drop = params["epochs_drop"], params["drop"]
        step = math.floor(epochs / epochs_drop)
        actual_lr = init_lr * math.pow(drop, step)
        key = (
            "learning_rate" if "learning_rate" in model.optimizer.get_config() else "lr"
        )
        learning_rate = model.optimizer.get_config().get(key, 0.0)
        assert np.isclose(
            actual_lr, learning_rate, rtol=rtol, atol=atol, equal_nan=False
        )
    elif name == "EarlyStopping":
        assert callback.stopped_epoch == 6
    params = {"epochs_drop": 100, "drop": 0.5, "patience": 10, "min_delta": 1e-5}
    callback.set_tunable_parameters(**params)
    check_params_tunable(callback, params, rtol, atol)
