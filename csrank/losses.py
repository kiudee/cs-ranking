import tensorflow as tf
from keras import backend as K

from csrank.tensorflow_util import tensorify

__all__ = ['hinged_rank_loss', 'make_smooth_ndcg_loss', 'smooth_rank_loss',
           'plackett_luce_loss']


def identifiable(loss_function):
    def wrap_loss(y_true, y_pred):
        alpha = 1e-4
        ss = tf.reduce_sum(tf.square(y_pred), axis=1)
        return alpha * ss + loss_function(y_true, y_pred)

    return wrap_loss


@identifiable
def hinged_rank_loss(y_true, y_pred):
    y_true, y_pred = tensorify(y_true), tensorify(y_pred)
    mask = K.cast(K.greater(y_true[:, None] - y_true[:, :, None], 0),
                  dtype='float32')
    diff = y_pred[:, :, None] - y_pred[:, None]
    hinge = K.maximum(mask * (1 - diff), 0)
    n = K.sum(mask, axis=(1, 2))
    return K.sum(hinge, axis=(1, 2)) / n


@identifiable
def smooth_rank_loss(y_true, y_pred):
    y_true, y_pred = tensorify(y_true), tensorify(y_pred)
    mask = K.cast(K.greater(y_true[:, None] - y_true[:, :, None], 0),
                  dtype='float32')
    exped = K.exp(y_pred[:, None] - y_pred[:, :, None])
    result = K.sum(exped * mask, axis=[1, 2])
    return result / K.sum(mask, axis=(1, 2))


@identifiable
def plackett_luce_loss(y_true, s_pred):
    y_true = tf.cast(y_true, dtype='int32')
    s_pred = tf.cast(s_pred, dtype='float32')
    m = tf.shape(y_true)[1]
    raw_max = tf.reduce_max(s_pred, axis=1, keepdims=True)
    max_elem = tf.stop_gradient(tf.where(
        tf.is_finite(raw_max),
        raw_max,
        tf.zeros_like(raw_max)))
    exped = tf.exp(tf.subtract(s_pred, max_elem))
    masks = tf.greater_equal(y_true, tf.range(m)[:, None, None])
    tri = exped * tf.cast(masks, tf.float32)
    lse = tf.reduce_sum(tf.log(tf.reduce_sum(tri, axis=2)), axis=0)
    return lse - tf.reduce_sum(s_pred, axis=1)


def make_smooth_ndcg_loss(y_true, y_pred):
    y_true, y_pred = tensorify(y_true), tensorify(y_pred)
    n_objects = K.max(y_true) + 1.
    y_true_f = K.cast(y_true, 'float32')
    relevance = n_objects - y_true_f - 1.
    log_term = K.log(relevance + 2.) / K.log(2.)
    exp_relevance = K.pow(2., relevance) - 1.
    gains = exp_relevance / log_term

    # Calculate ideal dcg:
    idcg = K.sum(gains, axis=-1)

    # Calculate smoothed dcg:
    exped = K.exp(y_pred)
    exped = exped / K.sum(exped, axis=-1, keepdims=True)
    # toppred, toppred_ind = tf.nn.top_k(gains * exped, k)
    return 1 - K.sum(exped * gains, axis=-1) / idcg
