import math
import os

import numpy as np
import tensorflow as tf
from keras import Sequential
from keras import backend as K
from keras.layers import Dense
from keras.optimizers import SGD

from csrank.callbacks import LRScheduler, EarlyStoppingWithWeights
from csrank.tests.test_ranking import check_params_tunable

lr = 0.015
model = Sequential()
model.add(Dense(10, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer=SGD(lr=lr), loss='binary_crossentropy')
patience = 5
min_delta = 1e-2
early_stop = EarlyStoppingWithWeights(min_delta=min_delta, patience=patience)
epochs_drop = 5
drop = 0.9
lr_scheduler = LRScheduler(epochs_drop=epochs_drop, drop=drop)
callbacks = [lr_scheduler, early_stop]


def sigmoid(x):
    x = 1. / (1. + np.exp(-x))
    return x


def trivial_classification_problem():
    random_state = np.random.RandomState(123)
    x = random_state.randn(200, 2)
    w = random_state.rand(2)
    y_true = np.array((sigmoid(np.dot(x, w)) > 0.5), dtype=np.int64)
    return x, y_true


def test_callbacks():
    tf.set_random_seed(0)
    os.environ["KERAS_BACKEND"] = "tensorflow"
    np.random.seed(123)
    x, y = trivial_classification_problem()
    epochs = 30
    model.fit(x, y, epochs=epochs, callbacks=callbacks, validation_split=0.1)
    rtol = 1e-2
    atol = 5e-4
    step = math.floor(early_stop.stopped_epoch / epochs_drop)
    actual_lr = lr * math.pow(drop, step)
    n_lr = K.get_value(model.optimizer.lr)
    assert np.isclose(actual_lr, n_lr, rtol=rtol, atol=atol, equal_nan=False)
    params = {"epochs_drop": 100, "drop": 0.5}
    lr_scheduler.set_tunable_parameters(**params)
    check_params_tunable(lr_scheduler, params, rtol, atol)
    assert early_stop.stopped_epoch == 10
    params = {"patience": 10, "min_delta": 1e-5}
    early_stop.set_tunable_parameters(**params)
    check_params_tunable(early_stop, params, rtol, atol)
