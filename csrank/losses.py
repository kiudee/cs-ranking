from keras import backend as K

from .util import tensorify

__all__ = ['hinged_rank_loss', 'make_smooth_ndcg_loss', 'smooth_rank_loss']


def identifiable(loss_function):
    def wrap_loss(y_true, y_pred):
        alpha = 1e-10
        scores_i = K.sum(y_pred, axis=1)
        sum_of_scores = K.cast(y_pred.get_shape().as_list()[1] / 2,
            dtype='float32')
        deviation = K.sum(K.square(scores_i - sum_of_scores))
        return alpha * deviation + loss_function(y_true, y_pred)

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

    def logp(y_true, y_pred):
        # Y is our n_instances x n_classes tensor of permutations
        # F is n_instances x n_classes (n_classes many latent processes
        def logsumexp(x):
            masked = K.reshape(
                K.boolean_mask(exped, K.greater_equal(y_pred, K.cast(x, 'float32'))),
                [n, -1])
            return max_entry + K.log(K.reduce_sum(masked, axis=1))

        n, m = K.shape(y_true)[0], K.shape(y_true)[1]
        max_entry = K.reduce_max(y_true)
        exped = K.exp(y_true - max_entry)
        lse = K.map_fn(logsumexp, K.range(m), dtype='float32')
        return K.reduce_sum(y_true) - K.reduce_sum(lse)

# def make_smooth_ndcg_at_k_loss(k=5):
#     def ndcg(y_true, y_pred):
#         n_objects = K.max(y_true) + 1.
#         y_true_f = K.cast(y_true, 'float32')
#         relevance = n_objects - y_true_f - 1.
#         log_term = K.log(relevance + 2.) / K.log(2.)
#         exp_relevance = K.pow(2., relevance) - 1.
#         gains = exp_relevance / log_term
#
#         # Calculate ideal dcg:
#         #toprel, toprel_ind = tf.nn.top_k(gains, k)
#         #idcg = K.sum(gains, axis=-1, keepdims=True)
#
#         # Calculate smoothed dcg:
#         exped = K.exp(y_pred)
#         exped = exped / K.sum(exped, axis=-1, keepdims=True)
#         toppred, toppred_ind = tf.nn.top_k(gains * exped, k)
#         return K.sum(toppred, axis=-1)
#
#     return ndcg
