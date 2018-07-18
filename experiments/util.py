from sklearn.metrics import hamming_loss, zero_one_loss

from csrank import *
from csrank.callbacks import EarlyStoppingWithWeights, LRScheduler, DebugOutput
from csrank.constants import *
from csrank.metrics import zero_one_rank_loss, zero_one_accuracy, make_ndcg_at_k_loss
from csrank.metrics_np import *
from csrank.objectranking.fate_object_ranker import FATEObjectRanker
from csrank.util import print_dictionary


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
    learner = learners[learner_name]
    learner = learner(**learner_params)
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
    hp_params['learner'] = learner
    hp_params['fit_params'] = fit_params
    hp_params['tunable_parameter_ranges'] = hp_ranges
    return hp_params


datasets = {SYNTHETIC_OR: ObjectRankingDatasetGenerator, DEPTH: DepthDatasetReader,
            SUSHI: SushiObjectRankingDatasetReader, IMAGE_DATASET: ImageDatasetReader,
            TAG_GENOME_OR: TagGenomeObjectRankingDatasetReader, SENTENCE_ORDERING: SentenceOrderingDatasetReader,
            LETOR_OR: LetorObjectRankingDatasetReader,
            LETOR_DC: LetorDiscreteChoiceDatasetReader, SYNTHETIC_DC: DiscreteChoiceDatasetGenerator,
            MNIST_DC: MNISTDiscreteChoiceDatasetReader, TAG_GENOME_DC: TagGenomeDiscreteChoiceDatasetReader,
            SYNTHETIC_CHOICE: ChoiceDatasetGenerator, MNIST_CHOICE: MNISTChoiceDatasetReader}
learners = {FETA_RANKER: FETAObjectRanker, RANKNET: RankNet, CMPNET: CmpNet, ERR: ExpectedRankRegression,
            RANKSVM: RankSVM, FATE_RANKER: FATEObjectRanker, LISTNET: ListNet,
            FETA_CHOICE: FETAChoiceFunction, FATE_CHOICE: FATEChoiceFunction}
try:
    from csrank import GeneralizedExtremeValueModel, FETADiscreteChoiceFunction, FATEDiscreteChoiceFunction, \
        RankNetDiscreteChoiceFunction, MultinomialLogitModel, NestedLogitModel, PairedCombinatorialLogit, \
        CmpNetDiscreteChoiceFunction, RankSVMDCM

    dcm_learners = {FETA_DC: FETADiscreteChoiceFunction, FATE_DC: FATEDiscreteChoiceFunction,
                    RANKNET_DC: RankNetDiscreteChoiceFunction, CMPNET_DC: CmpNetDiscreteChoiceFunction,
                    MNL: MultinomialLogitModel, NLM: NestedLogitModel, GEV: GeneralizedExtremeValueModel,
                    PCL: PairedCombinatorialLogit, RANKSVM_DC: RankSVMDCM}

except ImportError:
    dcm_learners = {}
    print('DCM models not implemented yet')

learners = {**learners, **dcm_learners}
print('Current learners \n', print_dictionary(learners))
ranking_metrics = {'KendallsTau': kendalls_mean_np, 'SpearmanCorrelation': spearman_scipy,
                   'ZeroOneRankLoss': zero_one_rank_loss_for_scores_np,
                   'ZeroOneRankLossTies': zero_one_rank_loss_for_scores_ties_np,
                   "ZeroOneAccuracy": zero_one_accuracy_np,
                   "NDCGTopAll": make_ndcg_at_k_loss_np}
discrete_choice_metrics = {'CategoricalAccuracy': categorical_accuracy_np,
                           'CategoricalTopK2': topk_categorical_accuracy_np(k=2),
                           'CategoricalTopK3': topk_categorical_accuracy_np(k=3),
                           'CategoricalTopK4': topk_categorical_accuracy_np(k=4),
                           'CategoricalTopK5': topk_categorical_accuracy_np(k=5),
                           'CategoricalTopK{}': topk_categorical_accuracy_np}
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
