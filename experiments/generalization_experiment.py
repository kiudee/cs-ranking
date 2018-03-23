"""Experiment runner for one dataset.

Usage:
  generalization_experiment.py (--n_objects=<n_objects>)

  generalization_experiment.py (-h | --help)

Arguments:
    FILE       An argument for passing in a file.
Options:
  -h --help             Show this screen.
  --n_objects=<n_objects>         No of Objects of the random generator [default: 5]
"""
import inspect
import os

import numpy as np
import pandas as pd
from docopt import docopt

from csrank.callbacks import DebugOutput
from csrank.dataset_reader import SyntheticDatasetGenerator
from csrank.fate_ranking import FATEObjectRanker
from csrank.metrics import zero_one_rank_loss_for_scores
from csrank.util import rename_file_if_exist, configure_logging_numpy_keras, get_tensor_value

MODEL = "aModel"

ERROR_OUTPUT_STRING = 'Out of sample error {} : {} for n_objects {}'


def generate_medoid_dataset(n_objects=5, random_state=42):
    parameters = {"n_features": 2, "n_objects": n_objects, "n_train_instances": 10000, "n_test_instances": 100000,
                  "dataset_type": "medoid", "random_state": random_state}
    generator = SyntheticDatasetGenerator(**parameters)
    return generator.get_single_train_test_split()


def get_evaluation_result(gor, X_train, Y_train, epochs):
    gor.fit(X_train, Y_train, callbacks=[DebugOutput()], epochs=epochs)
    eval_results = {}
    for n_objects in np.arange(3, 25):
        _, _, X_test, Y_test = generate_medoid_dataset(n_objects=n_objects, random_state=seed + n_objects * 5)
        y_pred_scores = gor.predict_scores(X_test, batch_size=X_test.shape[0])
        metric_loss = get_tensor_value(zero_one_rank_loss_for_scores(Y_test, y_pred_scores))
        logger.info(ERROR_OUTPUT_STRING.format("zero_one_rank_loss", str(np.mean(metric_loss)), n_objects))
        eval_results["n_test_objects {}".format(n_objects)] = metric_loss
    return eval_results


if __name__ == '__main__':
    arguments = docopt(__doc__)
    n_objects = int(arguments['--n_objects'])
    dirname = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    log_path = os.path.join(dirname, "logs", "generalizing_mean_{}.log".format(n_objects))
    df_path = os.path.join(dirname, "logs", "generalizing_mean_{}.csv".format(n_objects))
    log_path = rename_file_if_exist(log_path)
    df_path = rename_file_if_exist(df_path)
    random_state = np.random.RandomState(seed=42)
    seed = random_state.randint(2 ** 32)

    rows_list = []

    logger = configure_logging_numpy_keras(seed=seed, log_path=log_path)

    X_train, Y_train, _, _ = generate_medoid_dataset(n_objects=n_objects, random_state=seed)
    n_instances, n_objects, n_features = X_train.shape
    epochs = 1000
    params = {"n_objects": n_objects, "n_object_features": n_features, "use_early_stopping": True}

    logger.info("############################# With Default Set layers ##############################")
    gor = FATEObjectRanker(n_hidden_set_layers=2, n_hidden_set_units=2, **params)
    result = get_evaluation_result(gor, X_train, Y_train, epochs)
    result[MODEL] = "2SetLayersDefaultParams"
    rows_list.append(result)

    logger.info("############################# With 32 set layers ##############################")
    gor = FATEObjectRanker(n_hidden_set_layers=32, n_hidden_set_units=32, **params)
    result = get_evaluation_result(gor, X_train, Y_train, epochs)
    result[MODEL] = "32SetLayersDefaultParams"
    rows_list.append(result)

    logger.info("############################# With Best Parameters ##############################")
    best_point = [457, 300, 9.8565160071418972e-05, 2, 134, 10, 144, 8.6981334542701822e-05]
    gor = FATEObjectRanker(n_objects=n_objects, n_object_features=n_features)
    gor.set_tunable_parameter_ranges({})
    gor.set_tunable_parameters(best_point)
    result = get_evaluation_result(gor, X_train, Y_train, epochs)
    result[MODEL] = "10SetLayersBestParams"
    rows_list.append(result)

    logger.info("############################# With Best Parameters 2 ##############################")
    best_point = [64, 300, 0.00017720013951108102, 2, 251, 8, 127, 4.6538251716189899e-07]
    gor = FATEObjectRanker(n_objects=n_objects, n_object_features=n_features)
    gor.set_tunable_parameter_ranges({})
    gor.set_tunable_parameters(best_point)
    result = get_evaluation_result(gor, X_train, Y_train, epochs)
    result[MODEL] = "8SetLayersBestParams2"
    rows_list.append(result)

    df = pd.DataFrame(rows_list)
    df.to_csv(df_path)
    pd.read_csv(df_path)
