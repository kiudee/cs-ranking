"""Experiment runner for one dataset.

Usage:
  experiment_medoids.py (--seed=<seed> --iters=<iters> --cvfolds=<folds> --dataset=<dataset> --dataset_args=<dataset_args> --ranker_name=<ranker_name>  --problem=<problem> --duration=<duration>)

  experiment_medoids.py (-h | --help)

Arguments:
    FILE       An argument for passing in a file.
Options:
  -h --help             Show this screen.
  --seed=<seed>         Seed of the random generator [default: 123]
  --iters=<iters>       Number of iterations with optimizer [default: 50]
  --cvfolds=<folds>     Number of folds of the cross validation [default: 3]
  --dataset=<dataset>   The Dataset on which the the models will be optimized and trained [default: medoid]
  --dataset_args=<dataset_args> The dictionary arguments for the dataset on which the models will be optimized and trained [default: {}]
  --ranker=<ranker>     Type of ranker to be used for optimization [default: borda_zero]
  --problem=<problem>   Learning problem to be considered [default: 'object ranking']
  --duration=<duration> Duration of the complete experiment [default: 1D]
"""
import inspect
import os
import pickle as pk
from datetime import datetime

import h5py
import numpy as np
import pandas as pd
from docopt import docopt
from sklearn.model_selection import ShuffleSplit

from csrank.tuning import ParameterOptimizer
from csrank.util import create_dir_recursively, configure_logging_numpy_keras, \
    duration_tillnow, microsec_to_time, get_mean_loss_for_dictionary, \
    get_loss_for_array
from experiments.util import get_ranker_and_dataset_functions, get_ranker_parameters, ERROR_OUTPUT_STRING, \
    lp_metric_dict, get_duration_microsecond, get_applicable_ranker_dataset, get_dataset_str, log_test_train_data, \
    get_optimizer

DIR_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

OPTIMIZER_FOLDER = 'optimizer_single_fold'
FILE_FORMAT = '{}_{}'
LOGS_FOLDER = 'logs_single_fold'

if __name__ == '__main__':
    start = datetime.now()
    arguments = docopt(__doc__)
    seed = int(arguments['--seed'])
    n_iter = int(arguments['--iters'])
    n_folds = int(arguments['--cvfolds'])
    dataset_name = arguments['--dataset']
    dataset_function_params = arguments['--dataset_args']
    ranker_name = (arguments['--ranker_name'])
    problem = arguments['--problem']
    duration = arguments['--duration']

    duration = get_duration_microsecond(duration)

    dataset_name, ranker_name = get_applicable_ranker_dataset(dataset_name=dataset_name, ranker_name=ranker_name,
                                                              problem=problem)

    dataset_function_params, dataset_str = get_dataset_str(dataset_function_params, dataset_name)

    log_path = os.path.join(DIR_PATH, LOGS_FOLDER, (FILE_FORMAT + '.log').format(dataset_str, ranker_name))
    create_dir_recursively(log_path, True)
    # log_path = rename_file_if_exist(log_path)
    random_state = np.random.RandomState(seed=seed)

    logger = configure_logging_numpy_keras(seed=random_state.randint(2 ** 32), log_path=log_path)
    logger.debug(arguments)
    logger.debug("The duration {}".format(microsec_to_time(duration)))
    logger.debug("Started the experiment at {}".format(start))

    dataset_function_params["random_state"] = random_state
    ranker, dataset_reader = get_ranker_and_dataset_functions(ranker_name, dataset_name, dataset_function_params,
                                                              problem)

    X_train, Y_train, X_test, Y_test = dataset_reader.get_single_train_test_split()

    n_features, n_objects = log_test_train_data(X_train, X_test, logger)
    ranker_params, fit_params, parameter_ranges = get_ranker_parameters(ranker_name, n_features, n_objects,
                                                                        dataset_name,
                                                                        dataset_function_params)
    # fit_params["epochs"] = 5
    cv = ShuffleSplit(n_splits=n_folds, test_size=0.1, random_state=random_state)

    optimizer_path = os.path.join(DIR_PATH, OPTIMIZER_FOLDER, (FILE_FORMAT).format(dataset_str, ranker_name))
    df_path = os.path.join(DIR_PATH, "single_cv_results", (FILE_FORMAT + '.csv').format(dataset_str, "dataset"))
    if isinstance(Y_test, dict):
        pred_file = os.path.join(DIR_PATH, "predictions", (FILE_FORMAT + '.pkl').format(dataset_str, ranker_name))
    else:
        pred_file = os.path.join(DIR_PATH, "predictions", (FILE_FORMAT + '.h5').format(dataset_str, ranker_name))

    create_dir_recursively(df_path, is_file_path=True)
    create_dir_recursively(optimizer_path, True)
    create_dir_recursively(pred_file, is_file_path=True)

    optimizer, n_iter = get_optimizer(logger, optimizer_path, n_iter)
    if not (n_iter == 0 and os.path.isfile(pred_file)):
        optimizer_fit_params = {'n_iter': n_iter, 'cv_iter': cv, 'optimizer': optimizer,
                                "parameters_ranges": parameter_ranges, 'acq_func': 'EIps'}

        logger.debug('fit parameters: {}'.format(fit_params))
        logger.debug('ranker parameters: {}'.format(ranker_params))

        optimizer_params = {'ranker_class': ranker, "optimizer_path": optimizer_path, 'fit_params': fit_params,
                            'ranker_params': ranker_params,
                            'random_state': random_state}
        optimizer_model = ParameterOptimizer(**optimizer_params)

        time_taken = duration_tillnow(start)
        logger.info("Time Taken till now: {} milliseconds".format(microsec_to_time(time_taken)))
        # TODO Check this and make this according to the model
        time_spare_eout_eval = get_duration_microsecond('6H')
        logger.info("Time spared for the out of sample evaluation : {} ".format(microsec_to_time(time_spare_eout_eval)))

        optimizer_fit_params["total_duration"] = duration - time_taken - time_spare_eout_eval

        optimizer_model.fit(X_train, Y_train, **optimizer_fit_params)

        if optimizer_model.model is None:
            # Finally, fit a model on the complete training set:
            optimizer_model.model = optimizer_model._ranker_class(random_state=optimizer_model.random_state,
                                                                  **optimizer_model._ranker_params)
            best_point = optimizer_model.optimizer.Xi[np.argmin(optimizer_model.optimizer.yi)]
            optimizer_model.model.set_tunable_parameters(best_point)
            logger.info(optimizer_model.model.__dict__)
            optimizer_model.model.fit(X_train, Y_train, **optimizer_model._fit_params)

        if isinstance(X_test, dict):
            batch_size = 10000
        else:
            batch_size = X_test.shape[0]

        y_pred_scores = optimizer_model.predict_scores(X_test)
        del optimizer_model, X_train, Y_train, X_test
        if os.path.isfile(pred_file):
            os.remove(pred_file)
        if isinstance(y_pred_scores, dict):
            f = open(pred_file, "wb")
            pk.dump(y_pred_scores, f)
            f.close()
        else:
            f = h5py.File(pred_file, 'w')
            f.create_dataset('scores', data=y_pred_scores)
            f.close()
    else:
        logger.info("Predictions done evaluating the model")
        if '.pkl' in pred_file:
            f = open(pred_file, "rb")
            y_pred_scores = pk.load(f)
            f.close()
        else:
            f = h5py.File(pred_file, 'r')
            y_pred_scores = np.array(f['scores'])
            f.close()

    data = []
    one_row = ["{}".format(ranker_name.upper())]
    columns = ["Ranker"] + list(lp_metric_dict[problem].keys())
    for name, evaluation_metric in lp_metric_dict[problem].items():
        if isinstance(Y_test, dict):
            metric_loss = get_mean_loss_for_dictionary(logger, evaluation_metric, Y_test, y_pred_scores)
        else:
            metric_loss = get_loss_for_array(evaluation_metric, Y_test, y_pred_scores)
        logger.info(ERROR_OUTPUT_STRING % (name, metric_loss))
        one_row.append(metric_loss)
    data.append(one_row)

    if not os.path.isfile(df_path):
        dataFrame = pd.DataFrame(data, columns=columns)
    else:
        dataFrame = pd.read_csv(df_path)
        df2 = pd.DataFrame(data, columns=columns)
        dataFrame = dataFrame.append(df2, ignore_index=True)
    dataFrame.to_csv(df_path, index=False)
