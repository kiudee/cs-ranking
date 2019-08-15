LABEL_RANKING = 'label_ranking'
DYAD_RANKING = 'dyad_ranking'
OBJECT_RANKING = 'object_ranking'
DISCRETE_CHOICE = "discrete_choice"
CHOICE_FUNCTION = "choice_function"
EXCEPTION_OUTPUT_FEATURES_INSTANCES = "Number of instances inconsistent for {} dataset! output array:{} instances " \
                                      "and features arrays:{} instances"
EXCEPTION_OBJECT_ARRAY_SHAPE = "Invalid shape for {} dataset objects features array! shape is: {}"
EXCEPTION_CONTEXT_ARRAY_SHAPE = "Invalid shape for {} dataset context features array! shape is: {}"
EXCEPTION_UNWANTED_CONTEXT_FEATURES = "Unwanted extra context features in {} dataset"
EXCEPTION_RANKINGS_FEATURES_NO_OF_OBJECTS = "Number of objects inconsistent! in rankings:{} objects " \
                                            "and features array: {} objects"
EXCEPTION_RANKINGS = "Unwanted rankings in {} dataset"
EXCEPTION_SET_INCLUSION = "Choice Set inclusion/exclusion binary code not present for all the objects in the set."
allowed_dense_kwargs = ['input_shape', 'batch_input_shape', 'batch_size', 'dtype', 'name', 'trainable', 'weights',
                        'input_dtype', 'activation', 'use_bias', 'kernel_initializer', 'bias_initializer',
                        'kernel_regularizer', 'bias_regularizer', 'activity_regularizer', 'kernel_constraint',
                        'bias_constraint']
