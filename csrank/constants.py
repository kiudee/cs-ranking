LABEL_RANKING = 'label_ranking'
DYAD_RANKING = 'dyad_ranking'
OBJECT_RANKING = 'object_ranking'
DISCRETE_CHOICE = "discrete_choice"
CHOICE_FUNCTIONS = "choice_functions"
EXCEPTION_OUTPUT_FEATURES_INSTANCES = "Number of instances inconsistent for {} dataset! output array:{} instances " \
                                      "and features arrays:{} instances"
EXCEPTION_OBJECT_ARRAY_SHAPE = "Invalid shape for {} dataset objects features array! shape is: {}"
EXCEPTION_CONTEXT_ARRAY_SHAPE = "Invalid shape for {} dataset context features array! shape is: {}"
EXCEPTION_UNWANTED_CONTEXT_FEATURES = "Unwanted extra context features in {} dataset"
EXCEPTION_RANKINGS_FEATURES_NO_OF_OBJECTS = "Number of objects inconsistent! in rankings:{} objects " \
                                            "and features array: {} objects"
EXCEPTION_RANKINGS = "Unwanted rankings in {} dataset"
EXCEPTION_SET_INCLUSION = "Choice Set inclusion/exclusion binary code not present for all the objects in the set."

IMAGE_DATASET = 'image_dataset'
SUSHI = 'sushi'
SYNTHETIC_OR = 'synthetic_or'
SYNTHETIC_CHOICE = 'synthetic_choice'
SYNTHETIC_DC = 'synthetic_dc'
HYPER_VOLUME = "hyper_volume"
MNIST_CHOICE = "mnist_choice"
MNIST_DC = "mnist_dc"

DEPTH = 'depth'
SENTENCE_ORDERING = "sentence_ordering"
LETOR_OR = "letor_or"
LETOR_DC = "letor_dc"

RANKSVM = 'ranksvm'
ERR = 'err'
CMPNET = "cmpnet"
RANKNET = 'ranknet'
FETA_RANKER = 'feta_ranker'
FATE_RANKER = "fate_ranker"
LISTNET = 'listnet'
FETA_CHOICE = 'feta_choice'
FATE_CHOICE = "fate_choice"
TAG_GENOME_OR = 'tag_genome_or'
TAG_GENOME_DC = 'tag_genome_dc'


FETA_DC = 'feta_dc'
FATE_DC = "fate_dc"
RANKNET_DC = "ranknet_dc"
CMPNET_DC = "cmpnet_dc"
MNL = "multinomial_logit_model"
NLM = 'nested_logit_model'
GEV = 'generalized_extreme_value'
PCL = 'paired_combinatorial_logit'

allowed_dense_kwargs = ['input_shape', 'batch_input_shape', 'batch_size', 'dtype', 'name', 'trainable', 'weights',
                        'input_dtype', 'activation', 'use_bias', 'kernel_initializer', 'bias_initializer',
                        'kernel_regularizer', 'bias_regularizer', 'activity_regularizer', 'kernel_constraint',
                        'bias_constraint']
