import functools
import inspect
import itertools as iter
import logging
import multiprocessing
import os
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import seaborn as sns
import tensorflow as tf
from keras import backend as K
from keras.layers import Lambda
from matplotlib import pyplot as plt
from scipy.stats import spearmanr, rankdata
from sklearn.covariance import GraphLasso
from sklearn.utils import check_random_state
from tensorflow.python.client import device_lib

__all__ = ['scores_to_rankings', 'strongly_connected_components', 'create_graph_pairwise_matrix',
           'create_pairwise_prob_matrix', 'quicksort', 'get_rankings_tensor', 'get_instances_objects', 'tensorify',
           'deprecated', "print_dictionary", "configure_logging_numpy_keras",
           "create_input_lambda", 'files_with_same_name', 'rename_file_if_exist', "create_dir_recursively",
           'generate_seed', 'get_tensor_value', "spearman_mean_np", "zero_one_accuracy_np", "kendalls_mean_np",
           "zero_one_rank_loss_for_scores_ties_np", "normalize",
           "tunable_parameters_ranges", "zero_one_rank_loss_for_scores_np"]


def deprecated(func):
    """This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emmitted
    when the function is used.
    # Taken from https://wiki.python.org/moin/PythonDecoratorLibrary#Generating_Deprecation_Warnings
    """

    @functools.wraps(func)
    def new_func(*args, **kwargs):
        warnings.simplefilter('always', DeprecationWarning)  # turn off filter
        warnings.warn("Call to deprecated function {}.".format(func.__name__), category=DeprecationWarning,
                      stacklevel=2)
        warnings.simplefilter('default', DeprecationWarning)  # reset filter
        return func(*args, **kwargs)

    return new_func


def scores_to_rankings(score_matrix):
    mask3 = np.equal(score_matrix[:, None] - score_matrix[:, :, None], 0)
    n_objects = score_matrix.shape[1]
    ties = np.sum(np.sum(mask3, axis=(1, 2)) - n_objects)
    rankings = np.empty_like(score_matrix)
    if ties > 0:
        for i, s in enumerate(score_matrix):
            rankings[i] = rankdata(s) - 1
    else:
        orderings = np.argsort(score_matrix, axis=1)[:, ::-1]
        rankings = np.argsort(orderings, axis=1)
    return rankings


def ranking_ordering_conversion(input):
    output = np.argsort(input, axis=1)
    return output


def strongly_connected_components(graph):
    """ Find the strongly connected components in a graph using
        Tarjan's algorithm.
        # Taken from http://www.logarithmic.net/pfh-files/blog/01208083168/sort.py

        graph should be a dictionary mapping node names to
        lists of successor nodes.
        """

    result = []
    stack = []
    low = {}

    def visit(node):
        if node in low: return

        num = len(low)
        low[node] = num
        stack_pos = len(stack)
        stack.append(node)

        for successor in graph[node]:
            visit(successor)
            low[node] = min(low[node], low[successor])

        if num == low[node]:
            component = tuple(stack[stack_pos:])
            del stack[stack_pos:]
            result.append(component)
            for item in component:
                low[item] = len(graph)

    for node in graph:
        visit(node)

    return result


def create_graph_pairwise_matrix(matrix):
    n_objects = matrix.shape[0]
    graph = {key: [] for key in np.arange(n_objects)}
    for i, j in iter.combinations(np.arange(n_objects), 2):
        p_ij = matrix[i, j]
        p_ji = matrix[j, i]
        if (p_ij > p_ji):
            graph[j].append(i)
        if (p_ij < p_ji):
            graph[i].append(j)
        if (p_ij == p_ji):
            graph[j].append(i)
            graph[i].append(j)
    return graph


def create_pairwise_prob_matrix(n_objects):
    # Create a non-transitive pairwise probability matrix for n_objects*n_objects
    non_transitive = False
    while (not non_transitive):
        pairwise_prob = np.zeros([n_objects, n_objects])
        for i, j in iter.combinations(np.arange(n_objects), 2):
            pairwise_prob[i, j] = np.random.rand(1)[0]
            pairwise_prob[j, i] = 1.0 - pairwise_prob[i, j]
        for comp in strongly_connected_components(create_graph_pairwise_matrix(pairwise_prob)):
            if (len(comp) >= 3):
                non_transitive = True
                break
    return pairwise_prob


def quicksort(arr, matrix):
    # Apply the quick sort algorithm for the given set of objects and produces a ranking based on provided pairwise matrix
    if len(arr) < 2:
        return arr
    else:
        pivot = arr[0]
        right = [i for i in arr[1:] if matrix[pivot, i] == 1]
        left = [i for i in arr[1:] if matrix[pivot, i] == 0]
        return quicksort(left, matrix) + [pivot] + quicksort(right, matrix)


# Inputs are tensors
def get_rankings_tensor(n_objects, y_pred):
    # indices = orderings
    toprel, orderings = tf.nn.top_k(y_pred, n_objects)
    # indices = rankings
    troprel, rankings = tf.nn.top_k(orderings, n_objects)
    rankings = K.cast(rankings[:, ::-1], dtype='float32')
    return rankings


# Inputs are tensors
def get_instances_objects(y_true):
    n_objects = K.cast(K.int_shape(y_true)[1], 'int32')
    total = K.cast(K.greater_equal(y_true, 0), dtype='int32')
    n_instances = K.cast(tf.reduce_sum(total) / n_objects, dtype='int32')
    return n_instances, n_objects


def generate_seed(random_state=None):
    random_state = check_random_state(random_state)
    return random_state.randint(2 ** 32, dtype='uint32')


def tensorify(x):
    """Converts x into a Keras tensor"""
    if not isinstance(x, (tf.Tensor, tf.Variable)):
        return K.constant(x)
    return x


def get_tensor_value(x):
    if isinstance(x, tf.Tensor):
        return K.get_value(x)
    return x


def spearman_mean_np(y_true, y_pred):
    y_pred = scores_to_rankings(y_pred)
    rho = []
    for r1, r2 in zip(y_true, y_pred):
        rho.append(spearmanr(r1, r2)[0])
    return np.mean(np.array(rho))


def kendalls_mean_np(y_true, y_pred):
    return 1. - 2. * zero_one_rank_loss_for_scores_ties_np(y_true, y_pred)


def zero_one_accuracy_np(y_true, y_pred):
    y_pred = scores_to_rankings(y_pred)
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


def normalize(score):
    norm = [float(i) / np.sum(score) for i in score]
    norm = [float(i) / np.max(norm) for i in norm]
    return norm


def print_dictionary(dictionary):
    output = "\n"
    for key, value in dictionary.items():
        output = output + str(key) + " => " + str(value) + "\n"
    return output


def tunable_parameters_ranges(cls, logger, ranges):
    if (cls._tunable is None):
        cls.tunable_parameters()
    logger.info("Final tunable parameters with ranges {} ".format(print_dictionary(cls._tunable)))

    for name, new_range in ranges.items():
        if name in cls._tunable.keys():
            logger.info("New range for parameter: {} => {}".format(name, new_range))
            cls._tunable[name] = new_range
        else:
            logger.warning('This ranking algorithm does not support a tunable parameter called {}'.format(name))
    if len(ranges) > 0:
        logger.info("Customized tunable parameter ranges {}".format(print_dictionary(cls._tunable)))
    return list(cls._tunable.values())


def create_dir_recursively(path, is_file_path=False):
    if is_file_path:
        path = os.path.dirname(path)
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def rename_file_if_exist(file_path):
    my_file = Path(file_path)
    try:
        extension = '.' + file_path.split('.')[1]
    except IndexError:
        extension = ''
    path = file_path.split('.')[0]
    i = 1
    while my_file.is_file():
        file_path = path + str(i) + extension
        my_file = Path(file_path)
        i += 1
    return file_path


def files_with_same_name(file_path):
    files_list = []
    my_file = Path(file_path)
    try:
        extension = '.' + file_path.split('.')[1]
    except IndexError:
        extension = ''
    path = file_path.split('.')[0]
    i = 1
    while my_file.is_file():
        files_list.append(file_path)
        file_path = path + str(i) + extension
        my_file = Path(file_path)
        i += 1
    return files_list


def create_input_lambda(i):
    """Extracts off an object tensor from an input tensor"""
    return Lambda(lambda x: x[:, i])


def configure_logging_numpy_keras(seed=42, log_path=None, name='Experiment'):
    if log_path is None:
        dirname = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        log_path = os.path.join(dirname, "logs", "logs.log")
    logging.basicConfig(filename=log_path, level=logging.DEBUG,
                        format='%(asctime)s %(name)s %(levelname)-8s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger(name=name)

    tf.logging.set_verbosity(tf.logging.ERROR)
    logging.getLogger("tensorflow").setLevel(logging.ERROR)
    tf.set_random_seed(0)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["KERAS_BACKEND"] = "tensorflow"
    devices = [x.name for x in device_lib.list_local_devices()]
    logger.info("Devices {}".format(devices))
    n_gpus = len([x.name for x in device_lib.list_local_devices() if x.device_type == 'GPU'])
    if (n_gpus == 0):
        config = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1, \
                                allow_soft_placement=True, log_device_placement=False,
                                device_count={'CPU': multiprocessing.cpu_count() - 2})
    else:
        # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.50)
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)  # , gpu_options = gpu_options)
    sess = tf.Session(config=config)
    K.set_session(sess)
    np.random.seed(seed)
    logger.info("Number of GPUS {}".format(n_gpus))
    logger.info('tf session configuration: {}'.format(repr(sess._config)))
    logger.info("log file path: {}".format(log_path))
    return logger


def duration_tillnow(start):
    return (datetime.now() - start).total_seconds() * 1e6


def time_from_now(target_date_time_ms):
    base_datetime = datetime.now()
    delta = timedelta(0, 0, target_date_time_ms)
    target_date = base_datetime + delta
    return target_date.strftime("%Y-%m-%d %H:%M:%S")


def microsec_to_time(target_date_time_ms):
    return str(timedelta(microseconds=target_date_time_ms))


def get_mean_loss_for_dictionary(logger, metric, Y_true, Y_pred):
    losses = []
    total_instances = 0
    for n, y_pred in Y_pred.items():
        l = get_loss_for_array(metric, Y_true[n], y_pred)
        logger.info('nobj=>{} : loss/acc=>{} : ins=>{}'.format(n, l, y_pred.shape[0]))
        l = l * y_pred.shape[0]
        total_instances += y_pred.shape[0]
        losses.append(l)
    losses = np.array(losses)
    logger.info("total_instances {}".format(total_instances))
    logger.info(losses)
    loss = np.sum(losses) / total_instances
    return loss


def get_loss_for_dictionary(logger, metric, Y_true, Y_pred):
    losses = dict()
    total_instances = 0
    for n, y_pred in Y_pred.items():
        l = get_loss_for_array(metric, Y_true[n], y_pred)
        losses[n] = l
        logger.info('nobj=>{} : loss/acc=>{} : ins=>{}'.format(n, l, y_pred.shape[0]))
        # l = l * y_pred.shape[0]
        total_instances += y_pred.shape[0]
    logger.info("total_instances {}".format(total_instances))
    logger.info(print_dictionary(losses))
    return losses, total_instances


def get_loss_for_array(metric, y_true, y_pred):
    x = metric(y_true, y_pred)
    x = tensorify(x)
    return get_tensor_value(K.mean(x))


def heat_map(file_path, X, headers, cmap=sns.color_palette("Blues")):
    model = GraphLasso()
    model.fit(X)
    Cov = model.covariance_
    std = np.diag(1. / np.sqrt(np.diag(Cov)))
    Cor = std.dot(Cov).dot(std)

    fig, ax = plt.subplots()
    # the size of A4 paper
    fig.set_size_inches(10, 8)
    ax = sns.heatmap(Cor, cmap=cmap, square=True, xticklabels=1, yticklabels=1, linewidths=.5)
    ax.set_yticklabels(headers, rotation=0, fontsize=12)
    ax.set_xticklabels(headers, rotation=90, fontsize=12)
    plt.subplots_adjust(bottom=0.4, left=0.2)

    sns.despine(left=True, bottom=True)

    plt.tight_layout()

    plt.savefig(file_path)
    plt.show()