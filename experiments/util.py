import re
from collections import OrderedDict

import numpy as np
from keras.losses import categorical_crossentropy
from keras.metrics import categorical_accuracy
from keras.optimizers import SGD
from skopt import load

from csrank.callbacks import DebugOutput, LRScheduler
from csrank.constants import OBJECT_RANKING, LABEL_RANKING, DYAD_RANKING, DISCRETE_CHOICE, BATCH_SIZE, LEARNING_RATE, \
    LOG_UNIFORM
from csrank.dataset_reader import DepthDatasetReader, ImageDatasetReader, SushiObjectRankingDatasetReader, \
    SyntheticDatasetGenerator, TagGenomeDatasetReader, SentenceOrderingDatasetReader, LetorObjectRankingDatasetReader
from csrank.fate_ranking import FATEObjectRanker, N_HIDDEN_SET_LAYERS, N_HIDDEN_JOINT_LAYERS, \
    N_HIDDEN_SET_UNITS, N_HIDDEN_JOINT_UNITS
from csrank.losses import smooth_rank_loss
from csrank.metrics import zero_one_rank_loss_for_scores_ties, zero_one_rank_loss_for_scores, \
    spearman_correlation_for_scores, kendalls_tau_for_scores, zero_one_accuracy_for_scores
from csrank.objectranking.cmp_net import *
from csrank.objectranking.expected_rank_regression import ExpectedRankRegression
from csrank.objectranking.feta_ranker import FETANetwork
from csrank.objectranking.rank_net import RankNet
from csrank.objectranking.rank_svm import RankSVM
from csrank.util import kendalls_mean_np, spearman_mean_np, zero_one_accuracy_np, \
    zero_one_rank_loss_for_scores_ties_np, zero_one_rank_loss_for_scores_np

HYPER_VOLUME = "hyper_volume"

DEPTH = 'depth'

SENTENCE_ORDERING = "sentence_ordering"

LETOR = "letor"

RANKSVM = 'ranksvm'
ERR = 'err'
CMPNET = "cmpnet"
RANKNET = 'ranknet'
FETA_RANKER_ZERO = 'feta_ranker_zero'
FETA_RANKER = 'feta_ranker'
FATE_RANKER = "fate_ranker"

object_ranking_datasets = {'medoid': SyntheticDatasetGenerator, HYPER_VOLUME: SyntheticDatasetGenerator,
                           DEPTH: DepthDatasetReader,
                           'sushi': SushiObjectRankingDatasetReader, 'image_dataset': ImageDatasetReader,
                           'tag_genome': TagGenomeDatasetReader,
                           SENTENCE_ORDERING: SentenceOrderingDatasetReader,
                           LETOR: LetorObjectRankingDatasetReader}  # 'depth_semantic': generate_depth_dataset, 'depth_basic': generate_depth_dataset
dataset_arg_options = {HYPER_VOLUME: None, 'medoid': None, DEPTH: "dataset_type",
                       'sushi': None, 'image_dataset': None,
                       'tag_genome': None,
                       SENTENCE_ORDERING: ["n_dims", "train_obj"], LETOR: ["year", "train_obj"]}
object_rankers = {FETA_RANKER: FETANetwork, FETA_RANKER_ZERO: FETANetwork, RANKNET: RankNet, CMPNET: CmpNet,
                  ERR: ExpectedRankRegression, RANKSVM: RankSVM,
                  FATE_RANKER: FATEObjectRanker}  # , "bordaranknet_zero", "bordaranknet"

ranking_metrics = OrderedDict(
    {'KendallsTau': kendalls_tau_for_scores, 'SpearmanCorrelation': spearman_correlation_for_scores,
     'ZeroOneRankLoss': zero_one_rank_loss_for_scores,
     'ZeroOneRankLossTies': zero_one_rank_loss_for_scores_ties, "ZeroOneAccuracy": zero_one_accuracy_for_scores})
ranking_metrics_2 = OrderedDict(
    {'KendallsTau': kendalls_mean_np, 'SpearmanCorrelation': spearman_mean_np,
     'ZeroOneRankLoss': zero_one_rank_loss_for_scores_np,
     'ZeroOneRankLossTies': zero_one_rank_loss_for_scores_ties_np, "ZeroOneAccuracy": zero_one_accuracy_np})
discrete_choice_metrics = OrderedDict(
    {'categorical_accuracy': categorical_accuracy, 'CategoricalCrossEntropy': categorical_crossentropy})
lp_metric_dict = {
    OBJECT_RANKING: ranking_metrics_2,
    LABEL_RANKING: ranking_metrics,
    DYAD_RANKING: ranking_metrics,
    DISCRETE_CHOICE: discrete_choice_metrics
}
dataset_options_dict = {OBJECT_RANKING: (object_ranking_datasets)}
rankers_dict = {OBJECT_RANKING: object_rankers}
ERROR_OUTPUT_STRING = 'Out of sample error %s : %0.4f'


def get_ranker_and_dataset_functions(ranker_name, dataset_name, dataset_function_params, problem):
    rankers = rankers_dict[problem]
    datasets = dataset_options_dict[problem]
    ranker = rankers[ranker_name]
    dataset_reader = datasets[dataset_name]
    dataset_reader = dataset_reader(**dataset_function_params)
    return ranker, dataset_reader


def get_applicable_ranker_dataset(dataset_name, ranker_name, problem):
    rankers = rankers_dict[problem]
    datasets = dataset_options_dict[problem]
    if ranker_name not in rankers:
        ranker_name = FETA_RANKER_ZERO
    if dataset_name not in datasets:
        dataset_name = 'medoid'
    return dataset_name, ranker_name


def get_ranker_parameters(ranker_name, n_features, n_objects, dataset_name, dataset_function_params, epochs=1000):
    parameter_ranges = dict()

    ranker_params = {'n_objects': n_objects, "n_object_features": n_features, "n_features": n_features}
    if ranker_name == FETA_RANKER_ZERO or ranker_name == BORDA_RANKNET_ZERO:
        ranker_params['add_zeroth_order_model'] = True
    fit_params = {'epochs': epochs}
    fit_params['log_callbacks'] = []
    if dataset_name in [SENTENCE_ORDERING, HYPER_VOLUME, DEPTH, LETOR]:
        fit_params['log_callbacks'] = [LRScheduler()]
        parameter_ranges[BATCH_SIZE] = (512, 1024)
        parameter_ranges[LEARNING_RATE] = (1e-5, 1e-4, LOG_UNIFORM)
        ranker_params["optimizer"] = SGD(lr=1e-5, momentum=0.9, nesterov=True)

        if ranker_name in [FATE_RANKER, FETA_RANKER_ZERO, FETA_RANKER, BORDA_RANKNET_ZERO,
                           BORDA_RANKNET] and dataset_name not in [DEPTH,
                                                                   HYPER_VOLUME, LETOR]:
            ranker_params["loss_function"] = smooth_rank_loss

        if dataset_name == LETOR:
            parameter_ranges[LEARNING_RATE] = (1e-5, 1e-3, LOG_UNIFORM)
            parameter_ranges[BATCH_SIZE] = (1024, 2048)

        if dataset_name == DEPTH:
            parameter_ranges[LEARNING_RATE] = (1e-5, 1e-3, LOG_UNIFORM)
            parameter_ranges[BATCH_SIZE] = (256, 1024)
            if ranker_name in [FETA_RANKER_ZERO, FETA_RANKER, BORDA_RANKNET_ZERO, BORDA_RANKNET, RANKNET, CMPNET]:
                ranker_params["optimizer"] = "adam"
            parameter_ranges[N_HIDDEN_SET_LAYERS] = parameter_ranges[N_HIDDEN_JOINT_LAYERS] = (5, 20)
            parameter_ranges[N_HIDDEN_SET_UNITS] = parameter_ranges[N_HIDDEN_JOINT_UNITS] = (256, 1024)

    if ranker_name in [FETA_RANKER_ZERO, FETA_RANKER, FATE_RANKER, RANKNET, CMPNET, BORDA_RANKNET_ZERO, BORDA_RANKNET]:
        fit_params['log_callbacks'].append(DebugOutput())
        ranker_params["use_early_stopping"] = True
    if ranker_name == FATE_RANKER and dataset_name == SENTENCE_ORDERING and dataset_function_params.get(
            "train_obj", 0) == 0:
        fit_params = {'epochs': 35, "inner_epochs": 1, "min_bucket_size": 500, "validation_split": None}

    return ranker_params, fit_params, parameter_ranges


def get_duration_microsecond(duration):
    time = int(re.findall(r'\d+', duration)[0])
    d = duration.split(str(time))[1].upper()
    options = {"D": 24 * 60 * 60 * 1e6, "H": 60 * 60 * 1e6, "M": 60 * 1e6}
    return options[d] * time


def get_value(v):
    try:
        x = int(v.strip())
    except ValueError:
        x = str(v.strip())
    return x


def get_dataset_str(dataset_function_params, dataset_name):
    dataset_function_params = dict(
        (get_value(k), get_value(v)) for k, v in
        (item.split(':') for item in dataset_function_params.split(',')))

    keys = dataset_arg_options[dataset_name]
    if keys is not None:
        if isinstance(keys, list):
            string = ''
            for key in keys:
                if dataset_function_params.get(key, 0) != 0:
                    string = string + "{}_{}".format(key, dataset_function_params[key])
            dataset_str = "{}_{}".format(string, dataset_name)
        else:
            dataset_str = "{}_{}_{}".format(keys, str(dataset_function_params[keys]), dataset_name)
    else:
        dataset_str = dataset_name
    return dataset_function_params, dataset_str


def log_test_train_data(X_train, X_test, logger):
    if isinstance(X_train, dict) and isinstance(X_test, dict):
        n_instances, n_objects, n_features = X_train[list(X_train.keys())[0]].shape
        logger.info("instances {} objects {} features {}".format(n_instances, n_objects, n_features))
        logger.info("Using Test Set dictionary of rankings with lengths {}".format(X_test.keys()))
        logger.info("Using Training Set dictionary of rankings with lengths {}".format(X_train.keys()))
    if not isinstance(X_test, dict):
        n_i, n_o, n_f = X_test.shape
        logger.info("Test Set instances {} objects {} features {}".format(n_i, n_o, n_f))
    if not isinstance(X_train, dict):
        n_instances, n_objects, n_features = X_train.shape
        logger.info("Train Set instances {} objects {} features {}".format(n_instances, n_objects, n_features))
    return n_features, n_objects


def get_optimizer(logger, optimizer_path, n_iter):
    logger.info('Retrieving model stored at: {}'.format(optimizer_path))
    try:
        optimizer = load(optimizer_path)
        logger.info('Loading model stored at: {}'.format(optimizer_path))

    except KeyError:
        logger.error('Cannot open the file {}'.format(optimizer_path))
        optimizer = None

    except ValueError:
        logger.error('Cannot open the file {}'.format(optimizer_path))
        optimizer = None
    except FileNotFoundError:
        logger.error('No such file or directory: {}'.format(optimizer_path))
        optimizer = None
    if optimizer is not None:
        finished_iterations = np.array(optimizer.yi).shape[0]
        if finished_iterations == 0:
            optimizer = None
            logger.info('Optimizer did not finish any iterations so setting optimizer to null')
        else:
            n_iter = n_iter - finished_iterations
            if n_iter < 0:
                n_iter = 0
            logger.info('Iterations already done: {} and running iterations {}'.format(finished_iterations, n_iter))
    return optimizer, n_iter
