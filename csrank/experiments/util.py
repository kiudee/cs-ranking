import sys

import pymc3 as pm

from csrank.callbacks import EarlyStoppingWithWeights, LRScheduler, DebugOutput
from csrank.choicefunction import *
from csrank.constants import *
from csrank.dataset_reader import *
from csrank.discretechoice import *
from csrank.experiments.constants import *
from csrank.metrics import zero_one_rank_loss, zero_one_accuracy, make_ndcg_at_k_loss
from csrank.metrics_np import *
from csrank.objectranking import *

__all__ = ['log_test_train_data', 'get_dataset_reader', 'create_optimizer_parameters', 'create_optimizer_parameters2',
           'get_scores', 'datasets', 'cfs', 'ranking_metrics', 'discrete_choice_metrics', 'choice_metrics',
           'lp_metric_dict', 'metrics_on_predictions', 'callbacks_dictionary']

datasets = {SYNTHETIC_OR: ObjectRankingDatasetGenerator, DEPTH: DepthDatasetReader,
            SUSHI: SushiObjectRankingDatasetReader, IMAGE_DATASET: ImageDatasetReader,
            TAG_GENOME_OR: TagGenomeObjectRankingDatasetReader, SENTENCE_ORDERING: SentenceOrderingDatasetReader,
            LETOR_OR: LetorListwiseObjectRankingDatasetReader, SYNTHETIC_DC: DiscreteChoiceDatasetGenerator,
            LETOR_DC: LetorListwiseDiscreteChoiceDatasetReader, MNIST_DC: MNISTDiscreteChoiceDatasetReader,
            LETOR_RANKING_DC: LetorRankingDiscreteChoiceDatasetReader, SUSHI_DC: SushiDiscreteChoiceDatasetReader,
            TAG_GENOME_DC: TagGenomeDiscreteChoiceDatasetReader, SYNTHETIC_CHOICE: ChoiceDatasetGenerator,
            MNIST_CHOICE: MNISTChoiceDatasetReader, LETOR_CHOICE: LetorRankingChoiceDatasetReader,
            EXP_CHOICE: ExpediaChoiceDatasetReader, EXP_DC: ExpediaDiscreteChoiceDatasetReader}

cfs = {FETA_CHOICE: FETAChoiceFunction, FATE_CHOICE: FATEChoiceFunction, GLM_CHOICE: GeneralizedLinearModel,
       RANKNET_CHOICE: RankNetChoiceFunction, CMPNET_CHOICE: CmpNetChoiceFunction,
       FETALINEAR_CHOICE: FETALinearChoiceFunction, FATELINEAR_CHOICE: FATELinearChoiceFunction,
       RANKSVM_CHOICE: PairwiseSVMChoiceFunction, RANDOM_CHOICE: AllPositive}
ors = {FETA_RANKER: FETAObjectRanker, RANKNET: RankNet, CMPNET: CmpNet, ERR: ExpectedRankRegression,
       RANKSVM: RankSVM, FATE_RANKER: FATEObjectRanker, LISTNET: ListNet, FATELINEAR_RANKER: FATELinearObjectRanker,
       FETALINEAR_RANKER: FETALinearObjectRanker}
dcms = {FETA_DC: FETADiscreteChoiceFunction, FATE_DC: FATEDiscreteChoiceFunction,
        RANKNET_DC: RankNetDiscreteChoiceFunction, CMPNET_DC: CmpNetDiscreteChoiceFunction,
        MNL: MultinomialLogitModel, NLM: NestedLogitModel, GEV: GeneralizedNestedLogitModel,
        PCL: PairedCombinatorialLogit, RANKSVM_DC: PairwiseSVMDiscreteChoiceFunction, MLM: MixedLogitModel,
        FATELINEAR_DC: FATELinearDiscreteChoiceFunction, FETALINEAR_DC: FETALinearDiscreteChoiceFunction,
        RANDOM_DC: RandomBaselineDC}

learners = {**cfs, **dcms, **ors}

ranking_metrics = {'KendallsTau': kendalls_tau_for_scores_np,
                   'SpearmanCorrelation': spearman_correlation_for_scores_scipy,
                   'ZeroOneRankLoss': zero_one_rank_loss_for_scores_np,
                   'ZeroOneRankLossTies': zero_one_rank_loss_for_scores_ties_np,
                   "ZeroOneAccuracy": zero_one_accuracy_for_scores_np,
                   "NDCGTopAll": make_ndcg_at_k_loss_np}
discrete_choice_metrics = {'CategoricalAccuracy': categorical_accuracy_np,
                           'CategoricalTopK2': topk_categorical_accuracy_np(k=2),
                           'CategoricalTopK3': topk_categorical_accuracy_np(k=3),
                           'CategoricalTopK4': topk_categorical_accuracy_np(k=4),
                           'CategoricalTopK5': topk_categorical_accuracy_np(k=5),
                           'CategoricalTopK6': topk_categorical_accuracy_np(k=6)}
choice_metrics = {'F1Score': f1_measure, 'Precision': precision, 'Recall': recall,
                  'Subset01loss': subset_01_loss, 'HammingLoss': hamming, 'Informedness': instance_informedness,
                  "AucScore": auc_score, "AveragePrecisionScore": average_precision}
callbacks_dictionary = {'EarlyStoppingWithWeights': EarlyStoppingWithWeights, 'LRScheduler': LRScheduler,
                        'DebugOutput': DebugOutput, "CheckConvergence": pm.callbacks.CheckParametersConvergence,
                        "Tracker": pm.callbacks.Tracker}
lp_metric_dict = {
    OBJECT_RANKING: ranking_metrics,
    LABEL_RANKING: ranking_metrics,
    DYAD_RANKING: ranking_metrics,
    DISCRETE_CHOICE: discrete_choice_metrics,
    CHOICE_FUNCTION: choice_metrics
}
metrics_on_predictions = [f1_measure, precision, recall, subset_01_loss, hamming, instance_informedness,
                          zero_one_rank_loss, zero_one_accuracy, make_ndcg_at_k_loss, zero_one_accuracy_np]


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
    return n_objects


def get_dataset_reader(dataset_name, dataset_params):
    dataset_func = datasets[dataset_name]
    dataset_func = dataset_func(**dataset_params)
    return dataset_func


def create_optimizer_parameters(fit_params, hp_ranges, learner_params, learner_name, hash_file):
    hp_params = {}
    learner_func = learners[learner_name]
    learner = learner_func(**learner_params)
    learner.hash_file = hash_file
    if learner_name in hp_ranges.keys():
        hp_ranges[learner] = hp_ranges[learner_name]
        del hp_ranges[learner_name]
    if "callbacks" in fit_params.keys():
        callbacks = []
        for key, value in fit_params.get("callbacks", {}).items():
            callback = callbacks_dictionary[key]
            callback = callback(**value)
            callbacks.append(callback)
            if key in hp_ranges.keys():
                hp_ranges[callback] = hp_ranges[key]
                del hp_ranges[key]
        fit_params["callbacks"] = callbacks
    if "vi_params" in fit_params.keys():
        vi_params = fit_params["vi_params"]
        if "callbacks" in vi_params.keys():
            callbacks = []
            for key, value in vi_params.get("callbacks", {}).items():
                callback = callbacks_dictionary[key]
                callback = callback(**value)
                callbacks.append(callback)
            vi_params["callbacks"] = callbacks
        fit_params['vi_params'] = vi_params
    hp_params['learner'] = learner
    hp_params['fit_params'] = fit_params
    hp_params['tunable_parameter_ranges'] = hp_ranges
    return hp_params


def create_optimizer_parameters2(fit_params, hp_ranges, learner, learner_name, hash_file):
    hp_params = {}
    learner.hash_file = hash_file
    if learner_name in hp_ranges.keys():
        hp_ranges[learner] = hp_ranges[learner_name]
        del hp_ranges[learner_name]
    if "callbacks" in fit_params.keys():
        callbacks = []
        for key, value in fit_params.get("callbacks", {}).items():
            callback = callbacks_dictionary[key]
            callback = callback(**value)
            callbacks.append(callback)
            if key in hp_ranges.keys():
                hp_ranges[callback] = hp_ranges[key]
                del hp_ranges[key]
        fit_params["callbacks"] = callbacks
    if "vi_params" in fit_params.keys():
        vi_params = fit_params["vi_params"]
        if "callbacks" in vi_params.keys():
            callbacks = []
            for key, value in vi_params.get("callbacks", {}).items():
                callback = callbacks_dictionary[key]
                callback = callback(**value)
                callbacks.append(callback)
            vi_params["callbacks"] = callbacks
        fit_params['vi_params'] = vi_params
    hp_params['learner'] = learner
    hp_params['fit_params'] = fit_params
    hp_params['tunable_parameter_ranges'] = hp_ranges
    return hp_params


def get_scores(object, batch_size, X_test, Y_test, logger):
    s_pred = None
    while s_pred is None:
        try:
            if batch_size == 0:
                break
            logger.info("Batch_size {}".format(batch_size))
            if isinstance(object, AllPositive):
                s_pred = object.predict_scores(X_test, Y_test)
            else:
                s_pred = object.predict_scores(X_test, batch_size=batch_size)
        except:
            logger.error("Unexpected Error {}".format(sys.exc_info()[0]))
            s_pred = None
            batch_size = int(batch_size / 10)
    y_pred = object.predict_for_scores(s_pred)

    return s_pred, y_pred
