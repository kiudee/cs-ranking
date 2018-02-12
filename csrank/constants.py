LABEL_RANKING = 'label_ranking'
DYAD_RANKING = 'dyad_ranking'
OBJECT_RANKING = 'object_ranking'
DISCRETE_CHOICE = "discrete_choice"
EXCEPTION_RANKINGS_FEATURES_INSTANCES = "Number of instances inconsistent for {} dataset! rankings:{} instances and features arrays:{} instances"
EXCEPTION_OBJECT_ARRAY_SHAPE = "Invalid shape for {} dataset objects features array! shape is: {}"
EXCEPTION_CONTEXT_ARRAY_SHAPE = "Invalid shape for {} dataset context features array! shape is: {}"
EXCEPTION_UNWANTED_CONTEXT_FEATURES = "Unwanted extra context features in {} dataset"
EXCEPTION_RANKINGS_FEATURES_NO_OF_OBJECTS = "Number of objects inconsistent! in rankings:{} objects and features array: {} objects"

LOG_UNIFORM = 'log-uniform'

BATCH_SIZE = 'batch_size'
BATCH_SIZE_DEFAULT_RANGE = (64, 1024)

REDUCE_LR_ON_PLATEAU_FACTOR = 'reduced_lr_on_plateau_factor'
REDUCE_LR_ON_PLATEAU_FACTOR_DEFAULT_RANGE = (0.01, 0.7)
RLROP_DEFAULT_VALUE = 0.2

EARLY_STOPPING_PATIENCE = "early_stopping_patience"
EARLY_STOPPING_PATIENCE_DEFAULT_RANGE = (100, 500)
LEARNING_RATE = 'learning_rate'
LR_DEFAULT_RANGE = (1e-5, 1e-2, LOG_UNIFORM)

REGULARIZATION_FACTOR = 'regularization_factor'
REGULARIZATION_FACTOR_DEFAULT_RANGE = (1e-10, 1e-1, LOG_UNIFORM)
