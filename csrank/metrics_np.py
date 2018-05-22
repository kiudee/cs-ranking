import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, average_precision_score

from csrank.util import scores_to_rankings

__all__ = ['spearman_mean_np', 'kendalls_mean_np', 'zero_one_accuracy_np',
           'zero_one_rank_loss_for_scores_ties_np', 'zero_one_rank_loss_for_scores_np',
           'auc_score', "instance_informedness", 'f1_measure', 'recall',
           'average_precision', "precision"]


def spearman_mean_np(y_true, s_pred):
    y_pred = scores_to_rankings(s_pred)
    rho = []
    n_objects = y_true.shape[1]
    denominator = n_objects * (n_objects ** 2 - 1)

    for r1, r2 in zip(y_true, y_pred):
        if len(np.unique(r2)) == len(r2):
            s = 1 - (6 * np.sum((r1 - r2) ** 2) / denominator)
            rho.append(s)
        else:
            rho.append(np.nan)
    return np.nanmean(np.array(rho))


def spearman_scipy(y_true, s_pred):
    y_pred = scores_to_rankings(s_pred)
    rho = []
    for r1, r2 in zip(y_true, y_pred):
        s = spearmanr(r1, r2)[0]
        rho.append(s)
    return np.nanmean(np.array(rho))


def kendalls_mean_np(y_true, s_pred):
    return 1. - 2. * zero_one_rank_loss_for_scores_ties_np(y_true, s_pred)


def zero_one_accuracy_np(y_true, s_pred):
    y_pred = scores_to_rankings(s_pred)
    acc = np.sum(np.all(np.equal(y_true, y_pred), axis=1)) / y_pred.shape[0]
    return acc


def zero_one_rank_loss_for_scores_ties_np(y_true, s_pred):
    n_objects = y_true.shape[1]
    mask = np.greater(y_true[:, None] - y_true[:, :, None], 0).astype(float)
    mask2 = np.greater(s_pred[:, None] - s_pred[:, :, None], 0).astype(float)
    mask3 = np.equal(s_pred[:, None] - s_pred[:, :, None], 0).astype(float)

    # Calculate Transpositions
    transpositions = np.logical_and(mask, mask2)
    x = (np.sum(mask3, axis=(1, 2)) - n_objects).astype(float) / 4.0
    transpositions = np.sum(transpositions, axis=(1, 2)).astype(float)
    transpositions += x

    denominator = n_objects * (n_objects - 1.) / 2.
    result = transpositions / denominator
    return np.mean(result)


def zero_one_rank_loss_for_scores_np(y_true, s_pred):
    return zero_one_rank_loss_for_scores_ties_np(y_true, s_pred)


def auc_score(y_true, s_pred):
    try:
        auc = roc_auc_score(y_true, s_pred, average='samples')
    except ValueError:
        idx = np.where((y_true.sum(axis=1) != y_true.shape[-1]) & (y_true.sum(axis=1) != 1))
        auc = roc_auc_score(y_true[idx], s_pred[idx], average='samples')
    return auc


def average_precision(y_true, s_pred):
    return average_precision_score(y_true, s_pred, average='samples')


def instance_informedness(y_true, y_pred):
    tp = np.logical_and(y_true, y_pred).sum(axis=1)
    tn = np.logical_and(np.logical_not(y_true), np.logical_not(y_pred)).sum(axis=1)
    cp = y_true.sum(axis=1)
    cn = np.logical_not(y_true).sum(axis=1)
    return np.nanmean(tp / cp + tn / cn - 1)


def f1_measure(y_true, y_pred):
    return f1_score(y_true, y_pred, average='samples')


def precision(y_true, y_pred):
    return precision_score(y_true, y_pred, average='samples')


def recall(y_true, y_pred):
    return recall_score(y_true, y_pred, average='samples')


def categorical_topk_accuracy(y_true, y_pred, k=3):
    topK = y_pred.argsort(axis=1)[:, -k:][:, ::-1]
    accuracies = np.zeros_like(y_true, dtype=bool)
    for i, top in enumerate(topK):
        accuracies[i] = y_true[i] in top
    return np.mean(accuracies)

def categorical_accuracy(y_true, y_pred):
    pred = np.argmax(y_pred, axis=1)
    accuracies = np.equal(y_true, pred)
    return np.mean(accuracies)
