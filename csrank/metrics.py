import numpy as np
import tensorflow as tf
from keras import backend as K

from csrank.tensorflow_util import scores_to_rankings, get_instances_objects, tensorify

__all__ = ['zero_one_rank_loss', 'zero_one_rank_loss_for_scores', 'zero_one_rank_loss_for_scores_ties',
           'make_ndcg_at_k_loss', 'kendalls_tau_for_scores', 'spearman_correlation_for_scores', "zero_one_accuracy",
           "zero_one_accuracy_for_scores", "topk_categorical_accuracy", "point_dcg", "dcg", "ndcg"]


def zero_one_rank_loss(y_true, y_pred):
    y_true, y_pred = tensorify(y_true), tensorify(y_pred)
    mask = K.greater(y_true[:, None] - y_true[:, :, None], 0)
    # Count the number of mistakes (here position difference less than 0)
    mask2 = K.less(y_pred[:, None] - y_pred[:, :, None], 0)
    mask3 = K.equal(y_pred[:, None] - y_pred[:, :, None], 0)

    # Calculate Transpositions
    transpositions = tf.logical_and(mask, mask2)
    transpositions = K.sum(K.cast(transpositions, dtype='float32'), axis=[1, 2])

    n_objects = K.max(y_true) + 1
    transpositions += (K.sum(K.cast(mask3, dtype='float32'), axis=[1, 2])
                       - n_objects) / 4.
    denominator = K.cast((n_objects * (n_objects - 1.)) / 2., dtype='float32')
    result = transpositions / denominator
    return K.mean(result)


def zero_one_accuracy(y_true, y_pred):
    y_true, y_pred = tensorify(y_true), tensorify(y_pred)
    n_instances, n_objects = get_instances_objects(y_true)
    equal_ranks = K.cast(K.all(K.equal(y_pred, y_true), axis=1), dtype='float32')
    denominator = K.cast(n_instances, dtype='float32')
    zero_one_loss = K.sum(equal_ranks) / denominator
    return zero_one_loss


def zero_one_rank_loss_for_scores(y_true, s_pred):
    return zero_one_rank_loss_for_scores_ties(y_true, s_pred)


def zero_one_rank_loss_for_scores_ties(y_true, s_pred):
    y_true, s_pred = tensorify(y_true), tensorify(s_pred)
    n_objects = K.cast(K.max(y_true) + 1, dtype='float32')
    mask = K.greater(y_true[:, None] - y_true[:, :, None], 0)
    mask2 = K.greater(s_pred[:, None] - s_pred[:, :, None], 0)
    mask3 = K.equal(s_pred[:, None] - s_pred[:, :, None], 0)

    # Calculate Transpositions
    transpositions = tf.logical_and(mask, mask2)
    transpositions = K.sum(K.cast(transpositions, dtype='float32'), axis=[1, 2])
    transpositions += (K.sum(K.cast(mask3, dtype='float32'), axis=[1, 2])
                       - n_objects) / 4.

    denominator = n_objects * (n_objects - 1.) / 2.
    result = transpositions / denominator
    return K.mean(result)


def make_ndcg_at_k_loss(k=5):
    def ndcg(y_true, y_pred):
        y_true, y_pred = tensorify(y_true), tensorify(y_pred)
        n_objects = K.cast(K.int_shape(y_pred)[1], 'float32')
        relevance = K.pow(2., ((n_objects - y_true) * 60) / n_objects) - 1.
        relevance_pred = K.pow(2., ((n_objects - y_pred) * 60) / n_objects) - 1.

        # Calculate ideal dcg:
        toprel, toprel_ind = tf.nn.top_k(relevance, k)
        log_term = K.log(K.arange(k, dtype='float32') + 2.) / K.log(2.)
        idcg = K.sum(toprel / log_term, axis=-1, keepdims=True)
        # Calculate actual dcg:
        toppred, toppred_ind = tf.nn.top_k(relevance_pred, k)
        row_ind = K.cumsum(K.ones_like(toppred_ind), axis=0) - 1
        ind = K.stack([row_ind, toppred_ind], axis=-1)
        pred_rel = K.sum(tf.gather_nd(relevance, ind) / log_term, axis=-1,
                         keepdims=True)
        gain = pred_rel / idcg
        return gain

    return ndcg


def kendalls_tau_for_scores(y_true, y_pred):
    return 1. - 2. * zero_one_rank_loss_for_scores(y_true, y_pred)


def spearman_correlation_for_scores(y_true, y_pred):
    y_true, y_pred = tensorify(y_true), tensorify(y_pred)
    n_instances, n_objects = get_instances_objects(y_true)
    predicted_rankings = scores_to_rankings(n_objects, y_pred)
    y_true = K.cast(y_true, dtype='float32')
    sum_of_squared_distances = tf.constant(0.0)
    for i in np.arange(K.int_shape(y_pred)[1]):
        objects_pred = predicted_rankings[:, i]
        objects_true = y_true[:, i]
        t = (objects_pred - objects_true) ** 2
        sum_of_squared_distances = sum_of_squared_distances + tf.reduce_sum(t)
    denominator = K.cast(n_objects * (n_objects ** 2 - 1) * n_instances, dtype='float32')
    spearman_correlation = 1 - (6 * sum_of_squared_distances) / denominator
    return spearman_correlation


def zero_one_accuracy_for_scores(y_true, y_pred):
    y_true, y_pred = tensorify(y_true), tensorify(y_pred)
    n_instances, n_objects = get_instances_objects(y_true)
    predicted_rankings = scores_to_rankings(n_objects, y_pred)
    y_true = K.cast(y_true, dtype='float32')
    equal_ranks = K.cast(K.all(K.equal(predicted_rankings, y_true), axis=1), dtype='float32')
    denominator = K.cast(n_instances, dtype='float32')
    zero_one_loss = K.sum(equal_ranks) / denominator
    return zero_one_loss


def topk_categorical_accuracy(k=5):
    def topk_acc(y_true, y_pred):
        y_true, y_pred = tensorify(y_true), tensorify(y_pred)
        acc = tf.nn.in_top_k(y_pred, tf.argmax(y_true, axis=-1), k=k)
        acc = K.cast(acc, dtype='float32')
        return acc

    return topk_acc

def point_dcg(self, args):
    """
        Point DCG calculation function. Calculates the DCG for a given list. This list is assumed to be consisting of the rankings of documents belonging to the same query 
    """
        pos, label = args
        return (2 ** label - 1) / math.log(pos + 2, 2)

def dcg(self, preds):
    """
        List DCG calculation function. This function turns the list of rankings into a form which is easier to be passed to the point DCG function
    """
    return sum(map(self.point_dcg, enumerate(preds)))

def ndcg(self, preds, k=10):
    """
        NDCG calculation function that calculates the NDCG values with the help of the DCG calculation helper functions.
    """
    ideal_top = preds[:k]

    true_top = np.array([])
    if len(preds) > 10:
        true_top = np.partition(preds, -10)[-k:]
        true_top.sort()
    else:
        true_top = np.sort(preds)
    true_top = true_top[::-1]
    
    max_dcg = self.dcg(true_top)
    ideal_dcg = self.dcg(ideal_top)

    if max_dcg == 0:
        return 1

    return ideal_dcg / max_dcg