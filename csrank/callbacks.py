import logging
import math
import warnings

import numpy as np
from keras import backend as K
from keras.callbacks import Callback, EarlyStopping

from csrank.tunable import Tunable
from csrank.util import print_dictionary


class EarlyStoppingWithWeights(EarlyStopping, Tunable):
    """Stop training when a monitored quantity has stopped improving.
    # Arguments
        monitor: quantity to be monitored.
        min_delta: minimum change in the monitored quantity
            to qualify as an improvement, i.e. an absolute
            change of less than min_delta, will count as no
            improvement.
        patience: number of epochs with no improvement
            after which training will be stopped.
        verbose: verbosity mode.
        mode: one of {auto, min, max}. In `min` mode,
            training will stop when the quantity
            monitored has stopped decreasing; in `max`
            mode it will stop when the quantity
            monitored has stopped increasing; in `auto`
            mode, the direction is automatically inferred
            from the name of the monitored quantity.
    """

    def __init__(self, **kwargs):
        super(EarlyStoppingWithWeights, self).__init__(**kwargs)
        self.logger = logging.getLogger(EarlyStoppingWithWeights.__name__)

    def on_train_begin(self, logs=None):
        super(EarlyStoppingWithWeights, self).on_train_begin(logs=logs)
        self.epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        self.epoch += 1
        current = logs.get(self.monitor)
        self.best_weights = self.model.get_weights()
        if current is None:
            warnings.warn(
                'Early stopping conditioned on metric `%s` '
                'which is not available. Available metrics are: %s' %
                (self.monitor, ','.join(list(logs.keys()))), RuntimeWarning
            )
            return
        if self.monitor_op(current - self.min_delta, self.best):
            self.best = current
            self.best_weights = self.model.get_weights()
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = self.epoch
                self.model.stop_training = True

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            self.logger.info("Setting best weights for final epoch {}".format(self.epoch))
            self.model.set_weights(self.best_weights)

    def set_tunable_parameters(self, patience=300, min_delta=2, **point):
        self.patience = patience
        self.min_delta = min_delta
        if len(point) > 0:
            self.logger.warning('This callback does not support'
                                ' tunable parameters'
                                ' called: {}'.format(print_dictionary(point)))


class weightHistory(Callback):
    def on_train_begin(self, logs={}):
        self.zero_weights = []
        self.norm = []
        self.hidden_units_used = []

    def on_batch_end(self, batch, logs={}):
        hidden = [layer for layer in self.model.layers
                  if layer.name == 'hidden_1']

        y = np.array(hidden[0].get_weights()[0])
        close = np.isclose(y, 0, atol=1e-3)
        self.hidden_units_used.append(len(np.unique(np.where(np.logical_not(close))[1])))
        self.norm.append(np.abs(y).sum())
        self.zero_weights.append(close.sum())


class LRScheduler(Callback, Tunable):
    """Learning rate scheduler.

        Parameters
        ----------
        epochs_drop: unsigned int
            The number of epochs after which the learning rate is reduced
        drop: float [0,1):
            The percentage of the learning rate which needs to be dropped
        verbose: int. 0: quiet, 1: update messages#

    """

    def __init__(self, epochs_drop=300, drop=0.1, verbose=0, **kwargs):
        super(LRScheduler, self).__init__()
        self.verbose = verbose
        self.epochs_drop = epochs_drop
        self.drop = drop
        self.initial_lr = None

    def step_decay(self, epoch):
        step = math.floor((1 + epoch) / self.epochs_drop)
        new_lr = self.initial_lr * math.pow(self.drop, step)
        return new_lr

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')
        if epoch == 0:
            self.initial_lr = float(K.get_value(self.model.optimizer.lr))
        lr = self.step_decay(epoch)
        K.set_value(self.model.optimizer.lr, lr)
        if self.verbose > 0:
            print('\nEpoch %05d: LearningRateScheduler setting learning '
                  'rate to %s.' % (epoch + 1, lr))

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)

    def set_tunable_parameters(self, epochs_drop=300, drop=0.1, **point):
        self.epochs_drop = epochs_drop
        self.drop = drop
        if len(point) > 0:
            self.logger.warning('This callback does not support'
                                ' tunable parameters'
                                ' called: {}'.format(print_dictionary(point)))


class DebugOutput(Callback):

    def __init__(self, delta=100, **kwargs):
        """ Logging the epochs when done.

            Parameters
            ----------
            delta: unsigned int
                The number of epochs after which the message is logged
            kwargs:
                Keyword arguments

        """
        super(DebugOutput, self).__init__(**kwargs)
        self.delta = delta
        self.epoch = 0
        self.logger = logging.getLogger(DebugOutput.__name__)

    def on_train_end(self, logs=None):
        self.logger.debug('Total number of epochs: {}'.format(self.epoch))

    def on_train_begin(self, logs=None):
        self.epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        self.epoch += 1
        if self.epoch % self.delta == 0:
            self.logger.debug('Epoch {} of the training finished.'.format(self.epoch))
