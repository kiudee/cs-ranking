"""Experiment runner for full cluster cross validation on one dataset.

Usage:
  experiment_cv.py (--seed=<seed> --iters=<iters> --cvifolds=<cvifolds> --cvofolds=<cvofolds> --cindex=<cindex> --problem=<problem> --dataset=<dataset> --dataset_args=<dataset_args> --ranker_name=<ranker_name> --duration=<duration>)

  experiment_cv.py (-h | --help)

Arguments:
  FILE                  An argument for passing in a file.

Options:
  -h --help               Show this screen.
  --seed=<seed>           Seed of the random generator [default: 123]
  --iters=<iters>         Number of iterations with optimizer [default: 50]
  --cores=<cores>         Number of cores for the optimizer [default: 3]
  --cvifolds=<cvifolds>     Number of folds of the inner cross validation [default: 3]
  --cvofolds=<cvofolds>     Number of folds of the outer cross validation [default: 10]
  --cindex=<cindex>       Index given by the cluster to specify which job
                          is to be executed [default: 0]
  --problem=<problem>     Learning problem to be considered [default: 'object ranking']
  --dataset=<dataset>     The Dataset on which the the models will be optimized and trained [default: medoid]
  --dataset_args=<dataset_args> The dictionary arguments for the dataset on which the models will be optimized and trained [default: {}]
  --ranker_name=<ranker_name>       Type of ranker to be used for optimization [default: borda_zero]
  --duration=<duration> Duration of the complete experiment [default: 1D]
"""

import inspect
import os
from datetime import datetime

import numpy as np
import pandas as pd
from docopt import docopt
from sklearn.model_selection import ShuffleSplit
from skopt import load

from csrank.tuning import ParameterOptimizer
from csrank.util import (create_dir_recursively, configure_logging_numpy_keras,
                         microsec_to_time, get_mean_loss_for_dictionary,
                         get_loss_for_array)
from experiments.util import get_ranker_and_dataset_functions, get_ranker_parameters, ERROR_OUTPUT_STRING, \
    lp_metric_dict, get_duration_microsecond, get_applicable_ranker_dataset, get_dataset_str, \
    log_test_train_data

DIR_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

FILE_FORMAT = '{}_{}_{}'

LOGS_FOLDER = 'logs_multiple_folds'
OPTIMIZER_FOLDER = "optimizer_multiple_folds"

if __name__ == '__main__':
    start = datetime.now()
    arguments = docopt(__doc__)
    seed = int(arguments['--seed'])
    n_iter = int(arguments['--iters'])
    n_inner_folds = int(arguments['--cvifolds'])
    n_outer_folds = int(arguments['--cvofolds'])
    cluster_index = int(arguments['--cindex'])
    problem = arguments['--problem']
    dataset_name = arguments['--dataset']
    dataset_function_params = arguments['--dataset_args']
    ranker_name = (arguments['--ranker_name'])
    duration = arguments['--duration']
    duration = get_duration_microsecond(duration)
    dataset_name, ranker_name = get_applicable_ranker_dataset(dataset_name=dataset_name, ranker_name=ranker_name,
                                                              problem=problem)

    dataset_function_params, dataset_str = get_dataset_str(dataset_function_params, dataset_name)

    log_path = os.path.join(DIR_PATH, LOGS_FOLDER,
                            (FILE_FORMAT + '.log').format(dataset_str, ranker_name,
                                                          cluster_index))
    create_dir_recursively(log_path, True)
    # log_path = rename_file_if_exist(log_path)

    job_counter = 0
    for outercv in range(n_outer_folds):
        if job_counter == cluster_index:
            random_state = np.random.RandomState(seed=seed + outercv)
            logger = configure_logging_numpy_keras(seed=random_state.randint(2 ** 32, dtype='uint32'),
                                                   log_path=log_path)
            logger.debug(arguments)
            logger.debug("The duration in microseconds {}".format(duration))
            logger.debug("Starting the experiment at {}".format(start))
            dataset_function_params["random_state"] = random_state
            ranker, dataset_reader = get_ranker_and_dataset_functions(ranker_name, dataset_name,
                                                                      dataset_function_params, problem)

            X_train, Y_train, X_test, Y_test = dataset_reader.get_single_train_test_split()
            n_features, n_objects = log_test_train_data(X_train, X_test, logger)
            ranker_params, fit_params, parameter_ranges = get_ranker_parameters(ranker_name, n_features, n_objects,
                                                                                dataset_name,
                                                                                dataset_function_params)

            inner_cv = ShuffleSplit(n_splits=n_inner_folds, test_size=0.1, random_state=random_state)
            optimizer_path = os.path.join(DIR_PATH, OPTIMIZER_FOLDER,
                                          (FILE_FORMAT).format(dataset_str, ranker_name, cluster_index))
            create_dir_recursively(optimizer_path, True)
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
                    logger.info(
                        'Iterations already done: {} and running iterations {}'.format(finished_iterations, n_iter))

            optimizer_fit_params = {'n_iter': n_iter, 'cv_iter': inner_cv, 'optimizer': optimizer,
                                    "parameters_ranges": parameter_ranges, 'acq_func': 'EIps'}

            # fit_params['epochs'] = 5
            logger.debug('fit parameters: {}'.format(fit_params))
            logger.debug('ranker parameters: {}'.format(ranker_params))
            optimizer_params = {'ranker_class': ranker, 'fit_params': fit_params, 'ranker_params': ranker_params,
                                'random_state': random_state, 'optimizer_path': optimizer_path}
            optimizer_model = ParameterOptimizer(**optimizer_params)

            time_taken = (datetime.now() - start).microseconds
            logger.info("Time Taken till now: {}  milliseconds".format(microsec_to_time(time_taken)))
            time_spare_eout_eval = get_duration_microsecond('4H')
            logger.info(
                "Time spared for the out of sample evaluation : {} ".format(microsec_to_time(time_spare_eout_eval)))

            optimizer_fit_params["total_duration"] = duration - time_taken - time_spare_eout_eval
            optimizer_model.fit(X_train, Y_train, **optimizer_fit_params)

            if optimizer_model.model is None:
                logger.error('Final model was not fit on the complete training data.')
                optimizer_model.model = optimizer_model._ranker_class(
                    random_state=optimizer_model.random_state,
                    **optimizer_model._ranker_params)
                best_point = optimizer_model.optimizer.Xi[
                    np.argmin(optimizer_model.optimizer.yi)]
                optimizer_model.model.set_tunable_parameters(best_point)
                logger.info(optimizer_model.model.__dict__)
                optimizer_model.model.fit(X_train, Y_train,
                                          **optimizer_model._fit_params)

            if isinstance(X_test, dict):
                batch_size = 10000
            else:
                batch_size = X_test.shape[0]

            y_pred_rankings = optimizer_model.predict(X_test, batch_size=batch_size)
            y_pred_scores = optimizer_model.predict_scores(X_test, batch_size=batch_size)
            data = []
            one_row = ["fold_{}".format(cluster_index)]
            columns = ["Cluster"] + list(lp_metric_dict[problem].keys())
            for name, evaluation_metric in lp_metric_dict[problem].items():
                if isinstance(Y_test, dict):
                    metric_loss = get_mean_loss_for_dictionary(logger, evaluation_metric, Y_test, y_pred_scores)
                else:
                    metric_loss = get_loss_for_array(evaluation_metric, Y_test, y_pred_scores)
                logger.info(ERROR_OUTPUT_STRING % (name, metric_loss))
                one_row.append(metric_loss)
            data.append(one_row)
            df_path = os.path.join(DIR_PATH, "multiple_cv_results",
                                   ('{}_{}' + '.csv').format(dataset_str, ranker_name))
            create_dir_recursively(df_path, is_file_path=True)
            if not os.path.isfile(df_path):
                dataFrame = pd.DataFrame(data, columns=columns)
            else:
                dataFrame = pd.read_csv(df_path)
                df2 = pd.DataFrame(data, columns=columns)
                dataFrame = dataFrame.append(df2, ignore_index=True)
            dataFrame.to_csv(df_path, index=False)
        job_counter += 1
