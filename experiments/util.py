from sklearn.metrics import hamming_loss, zero_one_loss

from csrank import ObjectRankingDatasetGenerator, FETAObjectRanker, RankNet, CmpNet, ExpectedRankRegression, RankSVM, \
    FATEChoiceFunction, FETAChoiceFunction, DepthDatasetReader, SushiObjectRankingDatasetReader, ImageDatasetReader, \
    TagGenomeDatasetReader, SentenceOrderingDatasetReader, LetorObjectRankingDatasetReader, ChoiceDatasetGenerator, \
    MNISTChoiceDatasetReader, LetorDiscreteChoiceDatasetReader
from csrank.callbacks import EarlyStoppingWithWeights, LRScheduler, DebugOutput
from csrank.constants import SYNTHETIC_OR, DEPTH, SUSHI, IMAGE_DATASET, TAG_GENOME, SENTENCE_ORDERING, \
    LETOR_OR, FETA_RANKER, RANKNET, CMPNET, ERR, RANKSVM, FATE_RANKER, OBJECT_RANKING, LABEL_RANKING, DYAD_RANKING, \
    DISCRETE_CHOICE, CHOICE_FUNCTIONS, FETA_CHOICE, FATE_CHOICE, SYNTHETIC_CHOICE, MNIST_CHOICE, LETOR_DC
from csrank.metrics import zero_one_rank_loss, zero_one_accuracy, make_ndcg_at_k_loss
from csrank.metrics_np import *
from csrank.metrics_np import spearman_scipy, categorical_accuracy, categorical_topk_accuracy
from csrank.objectranking.fate_object_ranker import FATEObjectRanker


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
    dataset_reader = datasets[dataset_name]
    dataset_reader = dataset_reader(**dataset_params)
    return dataset_reader


def create_optimizer_parameters(fit_params, hp_ranges, learner_params, learner_name):
    hp_params = {}
    if learner_name in hp_ranges.keys():
        learner = learners[learner_name]
        learner = learner(**learner_params)
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
    hp_params['learner'] = learner
    hp_params['fit_params'] = fit_params
    hp_params['tunable_parameter_ranges'] = hp_ranges
    return hp_params


datasets = {SYNTHETIC_OR: ObjectRankingDatasetGenerator, DEPTH: DepthDatasetReader,
            SUSHI: SushiObjectRankingDatasetReader, IMAGE_DATASET: ImageDatasetReader,
            TAG_GENOME: TagGenomeDatasetReader, SENTENCE_ORDERING: SentenceOrderingDatasetReader,
            LETOR_OR: LetorObjectRankingDatasetReader,
            LETOR_DC: LetorDiscreteChoiceDatasetReader,
            SYNTHETIC_CHOICE: ChoiceDatasetGenerator, MNIST_CHOICE: MNISTChoiceDatasetReader}
learners = {FETA_RANKER: FETAObjectRanker, RANKNET: RankNet, CMPNET: CmpNet, ERR: ExpectedRankRegression,
            RANKSVM: RankSVM, FATE_RANKER: FATEObjectRanker, FETA_CHOICE: FETAChoiceFunction,
            FATE_CHOICE: FATEChoiceFunction}

ranking_metrics = {'KendallsTau': kendalls_mean_np, 'SpearmanCorrelation': spearman_scipy,
                   'ZeroOneRankLoss': zero_one_rank_loss_for_scores_np,
                   'ZeroOneRankLossTies': zero_one_rank_loss_for_scores_ties_np,
                   "ZeroOneAccuracy": zero_one_accuracy_np,
                   "NDCGTopAll": make_ndcg_at_k_loss}
discrete_choice_metrics = {'CategoricalAccuracy': categorical_accuracy, 'CategoricalTopK': categorical_topk_accuracy}
choice_metrics = {'F1Score': f1_measure, 'Precision': precision, 'Recall': recall,
                  'Subset01loss': zero_one_loss, 'HammingLoss': hamming_loss, 'Informedness': instance_informedness,
                  "AucScore": auc_score, "AveragePrecisionScore": average_precision}
callbacks_dictionary = {'EarlyStoppingWithWeights': EarlyStoppingWithWeights, 'LRScheduler': LRScheduler,
                        'DebugOutput': DebugOutput}
lp_metric_dict = {
    OBJECT_RANKING: ranking_metrics,
    LABEL_RANKING: ranking_metrics,
    DYAD_RANKING: ranking_metrics,
    DISCRETE_CHOICE: discrete_choice_metrics,
    CHOICE_FUNCTIONS: choice_metrics
}
ERROR_OUTPUT_STRING = 'Out of sample error %s : %0.4f'
metrics_on_predictions = [f1_measure, precision, recall, zero_one_loss, hamming_loss, instance_informedness,
                          zero_one_rank_loss, zero_one_accuracy, make_ndcg_at_k_loss]
