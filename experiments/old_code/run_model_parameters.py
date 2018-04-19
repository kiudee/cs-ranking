"""Experiment runner for one dataset.

Usage:
  run_model_parameters.py (--dataset_name=<dataset_name> --ranker_name=<ranker_name> --dataset_args=<dataset_args> --cindex=<cindex> --problem=<problem>)

  run_model_parameters.py (-h | --help)

Arguments:
    FILE       An argument for passing in a file.
Options:
  -h --help             Show this screen.
  --dataset_name=<dataset_name>   The Dataset name on which the the models will be optimized and trained [default: medoid]
  --ranker_name=<ranker_name>     Type of ranker name to be used for optimization [default: borda_zero]
  --dataset_args=<dataset_args> The dictionary arguments for the dataset on which the models will be optimized and trained [default: {}]
  --cindex=<cindex> The outer cross validation index [default: -1]
  --problem=<problem>     Learning problem to be considered [default: 'object ranking']

"""
import inspect
import os

import numpy as np
from docopt import docopt
from joblib import load

from csrank.util import configure_logging_numpy_keras, create_dir_recursively, get_mean_loss_for_dictionary, \
    get_loss_for_array
from experiments.util import get_ranker_and_dataset_functions, get_dataset_str, get_applicable_ranker_dataset, \
    log_test_train_data, get_ranker_parameters, lp_metric_dict, ERROR_OUTPUT_STRING

DIR_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

if __name__ == '__main__':
    arguments = docopt(__doc__)
    seed = 42
    dataset_name = arguments['--dataset_name']
    ranker_name = (arguments['--ranker_name'])
    dataset_function_params = arguments['--dataset_args']
    cindex = int(arguments['--cindex'])
    problem = arguments['--problem']
    dataset_name, ranker_name = get_applicable_ranker_dataset(dataset_name=dataset_name, ranker_name=ranker_name,
        problem=problem)
    dataset_function_params, dataset_str = get_dataset_str(dataset_function_params, dataset_name)
    file_name_format = '{}_{}'

    if cindex <= 0:
        folder = "{}_single_fold"
        result_folder = "single_cv_results"
    else:
        folder = "{}_multiple_folds"
        result_folder = "multiple_cv_results"
        file_name_format.format(file_name_format, cindex)

    log_path = os.path.join(DIR_PATH, folder.format('logs'), file_name_format.format(dataset_str, ranker_name) + '.log')

    random_state = np.random.RandomState(seed=seed)
    create_dir_recursively(log_path, True)
    # log_path = rename_file_if_exist(log_path)
    logger = configure_logging_numpy_keras(seed=random_state.randint(2 ** 32), log_path=log_path)
    logger.debug(arguments)
    dataset_function_params['random_state'] = random_state
    ranker, dataset_reader = get_ranker_and_dataset_functions(ranker_name, dataset_name, dataset_function_params,
        problem)

    X_train, Y_train, X_test, Y_test = dataset_reader.get_single_train_test_split()
    n_features, n_objects = log_test_train_data(X_train, X_test, logger)
    ranker_params, fit_params, parameter_ranges = get_ranker_parameters(ranker_name, n_features, n_objects,
        dataset_name,
        dataset_function_params)

    logger.info("ranker_params {} fit_params {}".format(ranker_params, fit_params))
    # Finally, fit a model on the complete training set:
    model = ranker(random_state=random_state, **ranker_params)
    model_path = os.path.join(DIR_PATH, folder.format('optimizer'), file_name_format.format(dataset_str, ranker_name))
    best_params = None
    try:
        opt = load(model_path)
        best_i = np.argmin(opt.yi)
        best_loss = opt.yi[best_i]
        best_params = opt.Xi[best_i]
        logger.info(
            'Best Params {} and loss {} for file {}'.format(best_params, best_loss, os.path.basename(model_path)))
    except KeyError:
        logger.error('Cannot open the file {}'.format(model_path))
    except ValueError:
        logger.error('Cannot open the file {}'.format(model_path))
    except FileNotFoundError:
        logger.error('File does not exist {}'.format(model_path))
    if best_params is None:
        exit()
    model.set_tunable_parameters(best_params)
    model.fit(X_train, Y_train, **fit_params)

    if 'batch_size' in fit_params:
        batch_size = fit_params['batch_size']
    else:
        batch_size = 32
    # X_test = X_test[10:20,:,:]
    # Y_test = Y_test[11:20, :]
    y_pred_scores = model.predict_scores(X_test, batch_size=batch_size)

    for name, evaluation_metric in lp_metric_dict[problem].items():
        if isinstance(Y_test, dict):
            metric_loss = get_mean_loss_for_dictionary(logger, evaluation_metric, Y_test, y_pred_scores)
        else:
            metric_loss = get_loss_for_array(evaluation_metric, Y_test, y_pred_scores)
        logger.info(ERROR_OUTPUT_STRING % (name, metric_loss))
    # Todo check the data frame code to save the results
