import logging
import multiprocessing
import os

from keras import backend as K
import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib


def scores_to_rankings(n_objects, y_pred):
    # indices = orderings
    toprel, orderings = tf.nn.top_k(y_pred, n_objects)
    # indices = rankings
    troprel, rankings = tf.nn.top_k(orderings, n_objects)
    rankings = K.cast(rankings[:, ::-1], dtype="float32")
    return rankings


def get_instances_objects(y_true):
    n_objects = K.cast(K.int_shape(y_true)[1], "int32")
    total = K.cast(K.greater_equal(y_true, 0), dtype="int32")
    n_instances = K.cast(tf.reduce_sum(total) / n_objects, dtype="int32")
    return n_instances, n_objects


def tensorify(x):
    """Converts x into a Keras tensor"""
    if not isinstance(x, (tf.Tensor, tf.Variable)):
        return K.constant(x)
    return x


def get_tensor_value(x):
    if isinstance(x, tf.Tensor):
        return K.get_value(x)
    return x


def configure_numpy_keras(seed=42):
    tf.set_random_seed(seed)
    os.environ["KERAS_BACKEND"] = "tensorflow"
    devices = [x.name for x in device_lib.list_local_devices()]
    logger = logging.getLogger("ConfigureKeras")
    logger.info("Devices {}".format(devices))
    n_gpus = len(
        [x.name for x in device_lib.list_local_devices() if x.device_type == "GPU"]
    )
    if n_gpus == 0:
        config = tf.ConfigProto(
            intra_op_parallelism_threads=1,
            inter_op_parallelism_threads=1,
            allow_soft_placement=True,
            log_device_placement=False,
            device_count={"CPU": multiprocessing.cpu_count() - 2},
        )
    else:
        config = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=True,
            intra_op_parallelism_threads=2,
            inter_op_parallelism_threads=2,
        )  # , gpu_options = gpu_options)
    sess = tf.Session(config=config)
    K.set_session(sess)
    np.random.seed(seed)
    logger.info("Number of GPUS {}".format(n_gpus))


def get_mean_loss(metric, y_true, y_pred):
    if isinstance(y_pred, dict) and isinstance(y_true, dict):
        losses = []
        total_instances = 0
        for n in y_pred.keys():
            loss = eval_loss(metric, y_true[n], y_pred[n])
            if not np.isnan(loss) and not np.isinf(loss):
                loss = loss * y_pred[n].shape[0]
                total_instances += y_pred[n].shape[0]
                losses.append(loss)
        losses = np.array(losses)
        mean_loss = np.sum(losses) / total_instances
    else:
        mean_loss = eval_loss(metric, y_true, y_pred)
    return mean_loss


def eval_loss(metric, y_true, y_pred):
    x = metric(y_true, y_pred)
    x = get_tensor_value(x)
    return np.nanmean(x)
