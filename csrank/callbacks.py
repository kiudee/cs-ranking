import logging
import math

from keras import backend as K
from keras.callbacks import Callback
import numpy as np

from csrank.tunable import Tunable
from csrank.util import print_dictionary


class EarlyStoppingWithWeights(Callback, Tunable):
    def __init__(
        self,
        monitor="val_loss",
        min_delta=0,
        patience=0,
        verbose=0,
        mode="auto",
        baseline=None,
        restore_best_weights=False,
        **kwargs,
    ):
        """Stop training when a monitored quantity has stopped improving.

        Parameters
        ----------
        monitor: string
            Quantity to be monitored, could be the loss or an accuracy value, which is monitored by the model.
            In case of accuracy the we check the change in increase, while in case of loss we check the change in
            decrease of the loss
        min_delta: float
            Minimum change in the monitored quantity to qualify as an improvement, i.e. an absolute change of less
            than min_delta, will count as no improvement
        patience: unsigned int
            number of epochs with no improvement after which training will be stopped
        verbose : bool
            verbosity mode, 1: print, 0 to not
        mode: one of {auto, min, max}.
            In `min` mode, training will stop when the quantity monitored has stopped decreasing; in `max` mode
            it will stop when the quantity monitored has stopped increasing; in `auto` mode, the direction is
            automatically inferred from the name of the monitored quantity
        baseline: float
            Baseline value for the monitored quantity to reach. Training will stop if the model doesn't show
            improvement over the baseline
        restore_best_weights: bool
            whether to restore model weights from the epoch with the best value of the monitored quantity.
            If False, the model weights obtained at the last step of training are used
        **kwargs
            Keyword arguments for the callback
        """
        super(EarlyStoppingWithWeights, self).__init__()
        self.monitor = monitor
        self.baseline = baseline
        self.patience = patience
        self.verbose = verbose
        self.min_delta = min_delta
        self.wait = 0
        self.stopped_epoch = 0
        self.restore_best_weights = restore_best_weights
        self.best_weights = None

        known_modes = {"auto", "min", "max"}
        if mode not in known_modes:
            raise ValueError(
                f"EarlyStopping mode {mode} is unknown, must be one of {known_modes}"
            )

        if mode == "min":
            self.monitor_op = np.less
        elif mode == "max":
            self.monitor_op = np.greater
        else:
            if "acc" in self.monitor:
                self.monitor_op = np.greater
            else:
                self.monitor_op = np.less

        if self.monitor_op == np.greater:
            self.min_delta *= 1
        else:
            self.min_delta *= -1
        self.logger = logging.getLogger(EarlyStoppingWithWeights.__name__)

    def on_train_begin(self, logs=None):
        # Allow instances to be re-used
        self.wait = 0
        self.stopped_epoch = 0
        if self.baseline is not None:
            self.best = self.baseline
        else:
            self.best = np.Inf if self.monitor_op == np.less else -np.Inf

    def on_epoch_end(self, epoch, logs=None):
        self.stopped_epoch += 1
        current = logs.get(self.monitor)
        self.best_weights = self.model.get_weights()
        if current is None:
            self.logger.warning(
                "Early stopping conditioned on metric `%s` which is not available. Available metrics are: %s"
                % (self.monitor, ",".join(list(logs.keys()))),
                RuntimeWarning,
            )
            return
        if self.monitor_op(current - self.min_delta, self.best):
            self.best = current
            self.best_weights = self.model.get_weights()
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.model.stop_training = True

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            self.logger.info(
                "Setting best weights for final epoch {}".format(self.stopped_epoch)
            )
            self.model.set_weights(self.best_weights)

    def set_tunable_parameters(self, patience=300, min_delta=2, **point):
        """Set tunable parameters of the EarlyStopping callback to the values provided.

        Parameters
        ----------
        patience: unsigned int
            Number of epochs with no improvement after which training will be stopped.
        min_delta: float
            Minimum change in the monitored quantity to qualify as an improvement, i.e. an absolute change of less
            than min_delta, will count as no improvement.
        point: dict
            Dictionary containing parameter values which are not tuned for the network
        """
        self.patience = patience
        self.min_delta = min_delta
        if len(point) > 0:
            self.logger.warning(
                "This callback does not support"
                " tunable parameters"
                " called: {}".format(print_dictionary(point))
            )


class LRScheduler(Callback, Tunable):
    def __init__(self, epochs_drop=300, drop=0.1, verbose=0, **kwargs):
        """Learning rate scheduler with step-decay function

        Parameters
        ----------
        epochs_drop: unsigned int
            The number of epochs after which the learning rate is reduced
        drop: float [0,1):
            The percentage of the learning rate which needs to be dropped
        verbose: bool or int in {0,1}
            int. 0: quiet, 1: update messages
        **kwargs
            Keyword arguments for the callback
        """
        super(LRScheduler, self).__init__(**kwargs)
        self.verbose = verbose
        self.epochs_drop = epochs_drop
        self.drop = drop
        self.initial_lr = None
        self.logger = logging.getLogger(LRScheduler.__name__)

    def step_decay(self, epoch):
        """The step-decay function, which takes the current epoch and update the learning rate according to the
        formulae.

        .. math::
            lr = lr_0 * d_r^{\\lfloor \\frac{e}{e_{\\text{drop}}}\\rfloor}

        where :math:`lr_0` is the learning rate at the zeroth epoch and :math:`0 < d_r < 1` is the rate with which
        the learning rate should be reduced, :math:`e` is the current epoch and :math:`e_{\\text{drop}}`
        is the number of epochs after which the learning rate is decreased.

        Parameters
        ----------
        epoch: unsigned int
            Current epoch
        """
        step = math.floor((1 + epoch) / self.epochs_drop)
        new_lr = self.initial_lr * math.pow(self.drop, step)
        return new_lr

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, "lr"):
            raise ValueError('Optimizer must have a "lr" attribute.')
        if epoch == 0:
            self.initial_lr = float(K.get_value(self.model.optimizer.lr))
        lr = self.step_decay(epoch)
        K.set_value(self.model.optimizer.lr, lr)
        if self.verbose > 0:
            print(
                "\nEpoch %05d: LearningRateScheduler setting learning "
                "rate to %s." % (epoch + 1, lr)
            )

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs["lr"] = K.get_value(self.model.optimizer.lr)

    def set_tunable_parameters(self, epochs_drop=300, drop=0.1, **point):
        """Set tunable parameters of the LRScheduler callback to the values provided.

        Parameters
        ----------
        epochs_drop: unsigned int
            The number of epochs after which the learning rate is reduced
        drop: float [0,1):
            The percentage of the learning rate which needs to be dropped
        point: dict
            Dictionary containing parameter values which are not tuned for the network
        """
        self.epochs_drop = epochs_drop
        self.drop = drop
        if len(point) > 0:
            self.logger.warning(
                "This callback does not support"
                " tunable parameters"
                " called: {}".format(print_dictionary(point))
            )


class DebugOutput(Callback):
    def __init__(self, delta=100, **kwargs):
        """Logging the epochs when done.

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
        self.logger.debug("Total number of epochs: {}".format(self.epoch))

    def on_train_begin(self, logs=None):
        self.epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        self.epoch += 1
        if self.epoch % self.delta == 0:
            self.logger.debug("Epoch {} of the training finished.".format(self.epoch))
