LABEL_RANKING = "label_ranking"
DYAD_RANKING = "dyad_ranking"
OBJECT_RANKING = "object_ranking"
DISCRETE_CHOICE = "discrete_choice"
CHOICE_FUNCTION = "choice_function"
EXCEPTION_OUTPUT_FEATURES_INSTANCES = (
    "Number of instances inconsistent for {} dataset! output array:{} instances "
    "and features arrays:{} instances"
)
EXCEPTION_OBJECT_ARRAY_SHAPE = (
    "Invalid shape for {} dataset objects features array! shape is: {}"
)
EXCEPTION_CONTEXT_ARRAY_SHAPE = (
    "Invalid shape for {} dataset context features array! shape is: {}"
)
EXCEPTION_UNWANTED_CONTEXT_FEATURES = "Unwanted extra context features in {} dataset"
EXCEPTION_RANKINGS_FEATURES_NO_OF_OBJECTS = (
    "Number of objects inconsistent! in rankings:{} objects "
    "and features array: {} objects"
)
EXCEPTION_RANKINGS = "Unwanted rankings in {} dataset"
EXCEPTION_SET_INCLUSION = "Choice Set inclusion/exclusion binary code not present for all the objects in the set."

RANKSVM = "ranksvm"

ERR = "err"
CMPNET = "cmpnet"
RANKNET = "ranknet"
FETA_RANKER = "feta_ranker"
FATE_RANKER = "fate_ranker"
LISTNET = "listnet"
FATELINEAR_RANKER = "fatelinear_ranker"
FETALINEAR_RANKER = "fetalinear_ranker"
RANDOM_RANKER = "random_ranker"

FETA_CHOICE = "feta_choice"
FETALINEAR_CHOICE = "fetalinear_choice"
FATE_CHOICE = "fate_choice"
FATELINEAR_CHOICE = "fatelinear_choice"
RANKNET_CHOICE = "ranknet_choice"
CMPNET_CHOICE = "cmpnet_choice"
RANKSVM_CHOICE = "ranksvm_choice"
GLM_CHOICE = "glm_choice"
RANDOM_CHOICE = "random_choice"

FETA_DC = "feta_dc"
FATE_DC = "fate_dc"
FATELINEAR_DC = "fatelinear_dc"
FETALINEAR_DC = "fetalinear_dc"
RANDOM_DC = "random_dc"

RANKNET_DC = "ranknet_dc"
CMPNET_DC = "cmpnet_dc"
MNL = "multinomial_logit_model"
NLM = "nested_logit_model"
GEV = "generalized_extreme_value"
PCL = "paired_combinatorial_logit"
RANKSVM_DC = "ranksvm_dc"
MLM = "mixed_logit_model"
