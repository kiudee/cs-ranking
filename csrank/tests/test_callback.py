import math

import numpy as np
from keras import Sequential
from keras import backend as K
from keras.layers import Dense
from keras.optimizers import SGD

from csrank.callbacks import LRScheduler

lr = 0.015
model = Sequential()
model.add(Dense(10, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer=SGD(lr=lr), loss='binary_crossentropy')


def sigmoid(x):
    x = 1. / (1. + np.exp(-x))
    return x


def trivial_classification_problem():
    random_state = np.random.RandomState(123)
    x = random_state.randn(200, 2)
    w = random_state.rand(2)
    y_true = np.array((sigmoid(np.dot(x, w)) > 0.5), dtype=np.int64)
    return x, y_true


def test_lr_scheduler():
    x, y = trivial_classification_problem()
    epochs_drop = 5
    drop = 0.9
    epochs = 10
    callbacks = [LRScheduler(epochs_drop=epochs_drop, drop=drop)]
    model.fit(x, y, epochs=epochs, callbacks=callbacks)
    rtol = 1e-2
    atol = 5e-2
    step = math.floor((epochs) / epochs_drop)
    actual_lr = lr * math.pow(drop, step)
    n_lr = K.get_value(model.optimizer.lr)
    assert np.isclose(actual_lr, n_lr, rtol=rtol, atol=atol, equal_nan=False)
