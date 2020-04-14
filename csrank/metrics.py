"""Various metrics that can be used to evaluate rankings.

All metrics take two parameters: `y_true` and `y_pred`. Both of these
are (n_instances, n_objects) shaped arrays of integers. We call these
arrays rankings. The element (i, j) of a ranking specifies the rank of
the jth object in the ith instance. `y_true` should be set to the
"ground truth" to evaluate against, while `y_pred` is the prediction
that should be evaluated.

Examples
--------
Lets assume we have two instances: ABCD and abcd. The "ground truth"
rankings are A > D > C > B and d < c < a < b.

We applied some ranking algorithm, which gave rankings of A > C > D > B
and d < a < b < c respectively.

Let's use some of the metrics defined here to evaluate the performance
of our ranker:

First encode the ground truth as a list of rankings. 0 is the highest
rank:
>>> y_true = [
...     [0, 3, 2, 1], # A > D > C > B, 0 is the highest rank
...     [2, 3, 1, 0], # d < c < a < b
... ]

Now similarly encode our prediction:
>>> y_pred = [
...     [0, 3, 1, 2], # A > C > D > B
...     [1, 2, 3, 0], # d < a < b < c
... ]

Evaluate with a simple zero-one loss:
>>> from keras import backend as K
>>> K.eval(zero_one_rank_loss(y_true, y_pred))
0.25

This is what we would expect: 25% of the objects were ranked at exactly the
right place. This might not be the most realistic metric, so let's try the
expected reciprocal rank instead:

>>> K.eval(err(y_true, y_pred))
0.6365559895833333
"""
from functools import partial

from keras import backend as K
import numpy as np
import tensorflow as tf

from csrank.tensorflow_util import get_instances_objects
from csrank.tensorflow_util import scores_to_rankings
from csrank.tensorflow_util import tensorify

__all__ = [
    "zero_one_rank_loss",
    "zero_one_rank_loss_for_scores",
    "zero_one_rank_loss_for_scores_ties",
    "make_ndcg_at_k_loss",
    "kendalls_tau_for_scores",
    "spearman_correlation_for_scores",
    "zero_one_accuracy",
    "zero_one_accuracy_for_scores",
    "topk_categorical_accuracy",
]


def zero_one_rank_loss(y_true, y_pred):
    y_true, y_pred = tensorify(y_true), tensorify(y_pred)
    mask = K.greater(y_true[:, None] - y_true[:, :, None], 0)
    # Count the number of mistakes (here position difference less than 0)
    mask2 = K.less(y_pred[:, None] - y_pred[:, :, None], 0)
    mask3 = K.equal(y_pred[:, None] - y_pred[:, :, None], 0)

    # Calculate Transpositions
    transpositions = tf.logical_and(mask, mask2)
    transpositions = K.sum(K.cast(transpositions, dtype="float32"), axis=[1, 2])

    n_objects = K.max(y_true) + 1
    transpositions += (
        K.sum(K.cast(mask3, dtype="float32"), axis=[1, 2]) - n_objects
    ) / 4.0
    denominator = K.cast((n_objects * (n_objects - 1.0)) / 2.0, dtype="float32")
    result = transpositions / denominator
    return K.mean(result)


def zero_one_accuracy(y_true, y_pred):
    y_true, y_pred = tensorify(y_true), tensorify(y_pred)
    n_instances, n_objects = get_instances_objects(y_true)
    equal_ranks = K.cast(K.all(K.equal(y_pred, y_true), axis=1), dtype="float32")
    denominator = K.cast(n_instances, dtype="float32")
    zero_one_loss = K.sum(equal_ranks) / denominator
    return zero_one_loss


def zero_one_rank_loss_for_scores(y_true, s_pred):
    return zero_one_rank_loss_for_scores_ties(y_true, s_pred)


def zero_one_rank_loss_for_scores_ties(y_true, s_pred):
    y_true, s_pred = tensorify(y_true), tensorify(s_pred)
    n_objects = K.cast(K.max(y_true) + 1, dtype="float32")
    mask = K.greater(y_true[:, None] - y_true[:, :, None], 0)
    mask2 = K.greater(s_pred[:, None] - s_pred[:, :, None], 0)
    mask3 = K.equal(s_pred[:, None] - s_pred[:, :, None], 0)

    # Calculate Transpositions
    transpositions = tf.logical_and(mask, mask2)
    transpositions = K.sum(K.cast(transpositions, dtype="float32"), axis=[1, 2])
    transpositions += (
        K.sum(K.cast(mask3, dtype="float32"), axis=[1, 2]) - n_objects
    ) / 4.0

    denominator = n_objects * (n_objects - 1.0) / 2.0
    result = transpositions / denominator
    return K.mean(result)


def make_ndcg_at_k_loss(k=5):
    r"""Computes the Normalized Discounted Cumulative Gain

    The Discounted Cumulative Gain is the sum of the document's relevancies,
    logarithmically discounted by their rank. That means the DCG is higher when
    the more relevant documents are highly ranked, lower otherwise.

    Concretely:

    .. math::
        \mathrm{DCG}_p = \sum_{i = 1}^p \frac{\mathit{rel}_i}{\log_2(i + 1)}

    Where :math:`\mathit{rel}_i` is the relevance of the document that is
    ranked at :math:`i`. Since this library deals with ranks, not relevances,
    it is necessary to define a conversion between the two. We define the
    relevance of an item as :math:`2^{\mathit{inv}}` where :math:`\mathit{inv}`
    is the negative rank normalized to :math:`[0, 1]`. An alternative way to
    view this is that :math:`\mathit{inv}` is the relevancy and our definition
    of ndcg exponentially discounts relevancies.

    To make the DCG comparable across different rankings (particularly rankings
    of different length), it is normalized by the ideal DCG. The resulting nDCG
    can be described as

    .. math::
        \mathrm{nDCG}_p = \frac{\mathrm{DCG}_p}{\mathrm{IDCG}_p}

    with

    .. math::
        \mathrm{IDCG}_p
        = \sum_{i = 1}^{\lvert \mathit{REL}_p \rvert} \frac{\mathit{rel}}{\log_2(i + 1)}

    where :math:`\mathit{REL}_p` is the list of relevant documents and
    :math:`\mathit{rel_i}` are the document relevancies in decreasing order.

    It follows that the nDCG is always a value in :math:`(0, 1]`, with
    :math:`1` being the best value.

    Parameters
    ----------
    k: int
        The length of the ranking for evaluation purposes. If the actual
        ranking is longer than `k`, only the (true) top `k` entries are
        considered. This is often more useful than considering the full
        ranking, for example when only a subset of the elements will actually
        be presented to a user.
    """

    def ndcg(y_true, y_pred):
        y_true, y_pred = tensorify(y_true), tensorify(y_pred)

        max_rank = K.max(y_true)

        def rank_to_relevance(rank):
            # Convert a rank to a relevance, which is (somewhat arbitrarily)
            # defined to be inversely proportional to the rank and normalized
            # to [0, 1]. Other conversion functions are possible.
            normalized_inverse = (max_rank - rank) / max_rank
            # define the relevance as 2**a
            return K.pow(2.0, normalized_inverse) - 1.0

        relevance_true = rank_to_relevance(y_true)
        relevance_pred = rank_to_relevance(y_pred)

        # Calculate ideal dcg:
        most_relevant_items, most_relevant_idx = tf.math.top_k(relevance_true, k)
        # arange starts at 0, but ranks start at 1 and the log term starts at 2
        log_term = K.log(K.arange(k, dtype="float32") + 2.0)
        # keras only natively supports the natural logarithm, have to switch base
        log2_term = log_term / K.log(2.0)
        idcg = K.sum(most_relevant_items / log2_term, axis=-1, keepdims=True)

        # Calculate actual dcg:

        # The index of the row of every element in toppred_ind, i.e.
        # [[0, 0],
        #  [1, 1]]
        row_ind = K.cumsum(K.ones_like(most_relevant_idx, dtype="int32"), axis=0) - 1

        # Indices of the k truly most relevant items, sorted by relevance. We
        # want to sort the predictions based on those indices, since that is
        # what we're trying to match.
        full_indices = K.stack([row_ind, most_relevant_idx], axis=-1)

        # Predicted relevances for the items that *should* have the top k
        # slots, ordered by the relevance rank they *should* have (so the
        # log2_term from the true predictions still has the right oder)
        top_k_preds = tf.gather_nd(relevance_pred, full_indices)

        weighted = top_k_preds / log2_term
        dcg = K.sum(weighted, axis=-1, keepdims=True)

        gain = dcg / idcg
        return gain

    return ndcg


def kendalls_tau_for_scores(y_true, y_pred):
    return 1.0 - 2.0 * zero_one_rank_loss_for_scores(y_true, y_pred)


def spearman_correlation_for_scores(y_true, y_pred):
    y_true, y_pred = tensorify(y_true), tensorify(y_pred)
    n_instances, n_objects = get_instances_objects(y_true)
    predicted_rankings = scores_to_rankings(n_objects, y_pred)
    y_true = K.cast(y_true, dtype="float32")
    sum_of_squared_distances = tf.constant(0.0)
    for i in np.arange(K.int_shape(y_pred)[1]):
        objects_pred = predicted_rankings[:, i]
        objects_true = y_true[:, i]
        t = (objects_pred - objects_true) ** 2
        sum_of_squared_distances = sum_of_squared_distances + tf.reduce_sum(t)
    denominator = K.cast(
        n_objects * (n_objects ** 2 - 1) * n_instances, dtype="float32"
    )
    spearman_correlation = 1 - (6 * sum_of_squared_distances) / denominator
    return spearman_correlation


def zero_one_accuracy_for_scores(y_true, y_pred):
    y_true, y_pred = tensorify(y_true), tensorify(y_pred)
    n_instances, n_objects = get_instances_objects(y_true)
    predicted_rankings = scores_to_rankings(n_objects, y_pred)
    y_true = K.cast(y_true, dtype="float32")
    equal_ranks = K.cast(
        K.all(K.equal(predicted_rankings, y_true), axis=1), dtype="float32"
    )
    denominator = K.cast(n_instances, dtype="float32")
    zero_one_loss = K.sum(equal_ranks) / denominator
    return zero_one_loss


def topk_categorical_accuracy(k=5):
    def topk_acc(y_true, y_pred):
        y_true, y_pred = tensorify(y_true), tensorify(y_pred)
        acc = tf.nn.in_top_k(y_pred, tf.argmax(y_true, axis=-1), k=k)
        acc = K.cast(acc, dtype="float32")
        return acc

    return topk_acc


def relevance_gain(grading, max_grade):
    """Maps a ranking (0 to `max_grade`, lower is better) to its gain.

    The gain is defined similar to the Discounted Cumulative Gain (DCG)
    metric. The value is always in [0, 1]. Therefore, it can be
    interpreted as a probability.

    Parameters
    ----------
    max_grade: float
        The highest achievable grade.
    grading: float
        A grading. Higher gradings are assumed to be better.
        0 <= grading <= max_grade must always hold.

    Tests
    -----
    >>> y_true = [0, 3, 2, 1] # (A > D > C > B)
    >>> K.eval(relevance_gain(y_true, max_grade=3)).tolist()
    [0.875, 0.0, 0.125, 0.375]
    """
    grading = tensorify(grading)
    inverse_grading = -grading + tf.cast(max_grade, grading.dtype)
    return (2 ** inverse_grading - 1) / (2 ** tf.cast(max_grade, tf.float32))


def err(y_true, y_pred, utility_function=None, probability_mapping=None):
    """Computes the Expected Reciprocal Rank or any Cascade Metric.

    ERR[1] is the cascade metric with the reciprocal rank as the utility
    function.

    Parameters
    ----------
    y_true: list of int
        The "ground truth" ranking. In this case, this does not need to
        actually be a ranking. It can be any 2 dimensional array whose
        elements can be transformed to probabilities by the
        `probability_mapping`.
    y_pred: list of int
        The predicted ranking that is to be evaluated.
    probability_mapping: list of int -> list of float
        A function that maps the elements of `y_true` to probabilities.
        Those values are then interpreted as the probability that the
        corresponding object satisfies the user's need. If `None` is
        specified, the `relevance_gain` function with `max_grade` set to
        the highest grade occurring in the grading is used.
    utility_function: int -> float
        A function that maps a rank (0 being the highest) to its
        "utility". If `None` is specified, this is defined as the
        reciprocal of the rank (resulting in the ERR metric). If a
        different utility is specified, this function can compute any
        cascade metric. Corresponds to the function represented by
        :math:`\\phi` in [1]. This will usually be a monotonically
        decreasing function, since the user is more likely to examine
        the first few results and therefore more likely to derive
        utility from them.

    Examples
    --------

    First, let's keep the default values and evaluate a ranking:
    >>> y_true = [
    ...     [0, 3, 2, 1],
    ...     [2, 3, 1, 0],
    ... ]
    >>> y_pred = [
    ...     [0, 3, 1, 2],
    ...     [1, 2, 3, 0],
    ... ]
    >>> K.eval(err(y_true, y_pred))
    0.6365559895833333

    Instead of relying on the relevance gain, we can also explicitly specify
    our own probabilities:
    >>> y_true = [
    ...     [0.3, 0.6, 0.05, 0.05],
    ...     [0.1, 0.1, 0.1, 0.7],
    ... ]
    >>> y_pred = [
    ...     [0, 3, 1, 2],
    ...     [1, 2, 3, 0],
    ... ]

    Now `y_true[i, j]` is the probability that object `j` in instance `i`
    satisfies the user's need. To use this probabilities unchanged, we need to
    override the probability mapping with the identity:

    >>> probability_mapping = lambda x: x

    Let us further specify that the rank utilities decrease in an exponential
    manner, e.g. every rank is only half as "valuable" as its predecessor:
    >>> utility_function = lambda r: 1/2**(r - 1) # start with 2**0 = 1

    We can now evaluate the metric:
    >>> K.eval(err(
    ...     y_true,
    ...     y_pred,
    ...     probability_mapping=probability_mapping,
    ...     utility_function=utility_function,
    ... ))
    0.3543499991945922

    The resulting metric is technically no longer an expected reciprocal rank,
    since the utility is not given by the reciprocal of the rank. It is a
    different version of a cascade metric. The original paper [1] called it an
    abandonment cascade (with gamma = 1/2), so let us define a new name for it:

    >>> from functools import partial
    >>> abandonment_cascade_half = partial(
    ...     err,
    ...     probability_mapping=probability_mapping,
    ...     utility_function=utility_function,
    ... )
    >>> K.eval(abandonment_cascade_half(y_true, y_pred))
    0.3543499991945922

    References
    ----------
        [1] Chapelle, Olivier, et al. "Expected reciprocal rank for graded
        relevance." Proceedings of the 18th ACM conference on Information and
        knowledge management. ACM, 2009. http://olivier.chapelle.cc/pub/err.pdf
    """
    if probability_mapping is None:
        max_grade = tf.reduce_max(y_true)
        probability_mapping = partial(relevance_gain, max_grade=max_grade)
    if utility_function is None:

        def reciprocal_rank(rank):
            return 1 / rank

        utility_function = reciprocal_rank
    y_true, y_pred = tensorify(y_true), tensorify(y_pred)

    ninstances = tf.shape(y_pred)[0]
    nobjects = tf.shape(y_pred)[1]

    # Using y_true and the probability mapping, we can derive the
    # probability that each object satisfies the users need (we need to
    # map over the flattened array and then restore the shape):
    satisfied_probs = tf.reshape(
        tf.map_fn(probability_mapping, tf.reshape(y_true, (-1,))), tf.shape(y_true)
    )

    # sort satisfied probabilities according to the predicted ranking
    rows = tf.range(0, ninstances)
    rows_cast = tf.broadcast_to(tf.reshape(rows, (-1, 1)), tf.shape(y_pred))
    full_indices = tf.stack([rows_cast, tf.cast(y_pred, tf.int32)], axis=2)
    satisfied_at_rank = tf.gather_nd(satisfied_probs, full_indices)

    not_satisfied_n_times = tf.cumprod(1 - satisfied_at_rank, axis=1, exclusive=True)

    # And from the positions predicted in y_pred we can further derive
    # the utilities of each object given their position:
    utilities = tf.map_fn(
        utility_function, tf.range(1, nobjects + 1), dtype=tf.float64,
    )

    discount_at_rank = tf.cast(not_satisfied_n_times, tf.float64) * tf.reshape(
        utilities, (1, -1)
    )
    discounted_document_values = (
        tf.cast(satisfied_at_rank, tf.float64) * discount_at_rank
    )
    results = tf.reduce_sum(discounted_document_values, axis=1)

    return K.mean(results)
