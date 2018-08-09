"""Experiment runner for one dataset.

Usage:
  performance_set_size.py (--dataset=<dataset> --dataset_type=<dataset_type> --runopt=<runopt>)

  performance_set_size.py (-h | --help)

Arguments:
    FILE       An argument for passing in a file.
Options:
  -h --help             Show this screen.
  --dataset=<dataset>         The dataset name
  --dataset_type=<dataset_type>   The dataset variant
  --runopt=<runopt> Boolean to run the optimizer
"""
import inspect
import logging
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from docopt import docopt
from sklearn.model_selection import ShuffleSplit
from skopt.utils import load

from csrank.constants import GEV, NLM, PCL, MNL
from csrank.metrics_np import categorical_accuracy_np
from csrank.tensorflow_util import configure_logging_numpy_keras
from csrank.util import print_dictionary, get_duration_seconds, duration_till_now, seconds_to_time
from experiments.dbconnection import DBConnector
from experiments.util import get_dataset_reader, learners, \
    callbacks_dictionary, create_optimizer_parameters, lp_metric_dict, ParameterOptimizer

OPTIMIZE_ON_OBJECTS = [5, 7, 15, 17]

N_OBJECTS_ARRAY = np.arange(3, 20)

MODEL = "aModel"

ERROR_OUTPUT_STRING = 'Out of sample error {} : {} for n_objects {}'

LOGS_FOLDER = 'logs'
OPTIMIZER_FOLDER = 'optimizers'
PREDICTIONS_FOLDER = 'predictions'
MODEL_FOLDER = 'models'
RESULT_FOLDER = 'results'

DIR_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))


def save_results(rows_list, df_path):
    df = pd.DataFrame(rows_list)
    df = df.set_index(MODEL).T
    df.insert(0, 'Objects', N_OBJECTS_ARRAY)
    df.to_csv(df_path, index=False)


def get_scores(object, batch_size):
    s_pred = None
    while s_pred is None:
        try:
            if batch_size == 0:
                break
            logger.info("Batch_size {}".format(batch_size))
            s_pred = object.predict_scores(X_test, batch_size=batch_size)
        except:
            logger.error("Unexpected Error {}".format(sys.exc_info()[0]))
            s_pred = None
            batch_size = int(batch_size / 10)
    y_pred = object.predict_for_scores(s_pred)

    return s_pred, y_pred


if __name__ == '__main__':
    arguments = docopt(__doc__)
    start = datetime.now()
    dataset = str(arguments['--dataset'])
    dataset_type = str(arguments['--dataset_type'])
    run_opt = bool(arguments['--runopt'])
    dirname = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    log_path = os.path.join(dirname, LOGS_FOLDER, "performance_sets_{}_{}.log".format(dataset, dataset_type))
    df_path = os.path.join(dirname, RESULT_FOLDER, "performance_sets_{}_{}.csv".format(dataset, dataset_type))
    config_file_path = os.path.join(DIR_PATH, 'config', 'clusterdb.json')

    configure_logging_numpy_keras(seed=42, log_path=log_path)
    dbConnector = DBConnector(config_file_path=config_file_path)
    dbConnector.init_connection()
    select_st = "SELECT * FROM {0} WHERE {0}.dataset=\'{1}\' AND dataset_params->>'dataset_type'=\'{2}\'".format(
        '{0}', dataset, dataset_type)
    models_done = []
    if os.path.isfile(df_path):
        df = pd.read_csv(df_path)
        models_done = list(df.columns)
    run_jobs = []
    dbConnector.cursor_db.execute(select_st.format('masterthesis.avail_jobs'))
    for job in dbConnector.cursor_db.fetchall():
        if job['fold_id'] == 0:
            run_jobs.append(dict(job))
    dbConnector.cursor_db.execute(select_st.format('pymc3.avail_jobs'))
    for job in dbConnector.cursor_db.fetchall():
        if job['fold_id'] == 0:
            run_jobs.append(dict(job))

    dbConnector.close_connection()
    rows_list = []
    duration = get_duration_seconds('10D')
    logger = logging.getLogger('PerformanceSetSizes')
    logger.info("DB config filePath {}".format(config_file_path))
    logger.info("Arguments {}".format(arguments))
    if dataset_type == 'median':
        N_OBJECTS_ARRAY = np.arange(3, 20, step=2)
        OPTIMIZE_ON_OBJECTS = [5, 15]
    for job_description in run_jobs:
        seed = int(job_description["seed"])
        job_id = int(job_description["job_id"])
        fold_id = int(job_description["fold_id"])
        dataset_name = job_description["dataset"]
        n_inner_folds = int(job_description["inner_folds"])
        dataset_params = job_description["dataset_params"]
        learner_name = job_description["learner"]
        fit_params = job_description["fit_params"]
        learner_params = job_description["learner_params"]
        hp_iters = int(job_description["hp_iters"])
        hp_ranges = job_description["hp_ranges"]
        hp_fit_params = job_description["hp_fit_params"]
        learning_problem = job_description["learning_problem"]
        experiment_schema = job_description["experiment_schema"]
        experiment_table = job_description["experiment_table"]
        validation_loss = job_description["validation_loss"]
        hash_value = job_description["hash_value"]
        random_state = np.random.RandomState(seed=seed + fold_id)

        configure_logging_numpy_keras(seed=seed, log_path=log_path)
        optimizer_path = os.path.join(DIR_PATH, OPTIMIZER_FOLDER, "{}".format(hash_value))
        hash_file = os.path.join(DIR_PATH, MODEL_FOLDER, "{}.h5".format(hash_value))
        logger.info("learner {} dataset {}".format(learner_name, dataset_name))
        dataset_params['random_state'] = random_state
        dataset_params['fold_id'] = fold_id
        dataset_reader = get_dataset_reader(dataset_name, dataset_params)
        inner_cv = ShuffleSplit(n_splits=n_inner_folds, test_size=0.1, random_state=random_state)
        if learner_name in [MNL, PCL, NLM, GEV]:
            fit_params['random_seed'] = seed + fold_id
        optimizer = load(optimizer_path)
        if "ps" in optimizer.acq_func:
            best_i = np.argmin(np.array(optimizer.yi)[:, 0])
        else:
            best_i = np.argmin(optimizer.yi)
        best_point = optimizer.Xi[best_i]
        best_loss = optimizer.yi[best_i]
        logger.info("Best parameters so far with a loss of {:.4f}:\n {}".format(best_loss, best_point))
        add_in_name = ''
        if job_description['learner_params'].get("add_zeroth_order_model", False):
            add_in_name = '_zero'
        model_name = '{}{}'.format(learner_name, add_in_name)

        if model_name not in models_done:
            X_train, Y_train, X_test, Y_test = dataset_reader.get_single_train_test_split()
            learner_params['n_objects'], learner_params['n_object_features'] = X_train.shape[1:]
            logger.info("learner params {}".format(print_dictionary(learner_params)))
            learner = learners[learner_name]
            learner = learner(**learner_params)
            learner.hash_file = hash_file
            tuned_objects = {learner: hp_ranges[learner_name].keys()}
            if "callbacks" in fit_params.keys():
                callbacks = []
                for key, value in fit_params.get("callbacks", {}).items():
                    callback = callbacks_dictionary[key]
                    callback = callback(**value)
                    callbacks.append(callback)
                    if key in hp_ranges.keys():
                        tuned_objects[callback] = hp_ranges[key].keys()
                fit_params["callbacks"] = callbacks
            logger.info('tuned objects {}'.format(print_dictionary(tuned_objects)))
            eval_results = {MODEL: model_name}
            for n_objects in N_OBJECTS_ARRAY:
                dataset_reader.n_objects = n_objects
                X_train, Y_train, X_test, Y_test = dataset_reader.get_single_train_test_split()
                learner.n_objects = Y_train.shape[-1]
                i = 0
                for obj, parameters in tuned_objects.items():
                    param_dict = dict()
                    for j, p in enumerate(parameters):
                        param_dict[p] = best_point[i + j]
                    logger.info('obj: {}, current parameters {}'.format(type(obj).__name__, param_dict))
                    obj.set_tunable_parameters(**param_dict)
                    i += len(parameters)
                learner.fit(X_train, Y_train, **fit_params)
                batch_size = X_test.shape[0]
                s_pred, y_pred = get_scores(learner, batch_size)
                metric_loss = categorical_accuracy_np(Y_test, y_pred)
                logger.info(ERROR_OUTPUT_STRING.format("CategoricalAccuracy", str(np.mean(metric_loss)), n_objects))
                eval_results[n_objects] = metric_loss
            rows_list.append(eval_results)
            save_results(rows_list, df_path)
        if run_opt:
            model_name_opt = '{}_optimized'.format(model_name)
            if model_name_opt not in models_done or learner_name not in [MNL, PCL]:
                logger.info("Checking results for optimized learner on {}".format([3, 5, 7, 13, 15]))
                eval_results = {MODEL: model_name_opt}
                for n in N_OBJECTS_ARRAY:
                    eval_results[n] = 0.0
                for n_objects in OPTIMIZE_ON_OBJECTS:
                    dataset_reader.n_objects = n_objects
                    X_train, Y_train, X_test, Y_test = dataset_reader.get_single_train_test_split()
                    learner_params['n_objects'], learner_params['n_object_features'] = X_train.shape[1:]
                    if 'n_nests' in hp_ranges[learner_name]:
                        hp_ranges[learner_name]['n_nests'] = [2, np.max([3, int(n_objects / 2) + 1])]
                    hp_params = create_optimizer_parameters(fit_params, hp_ranges, learner_params, learner_name,
                                                            hash_file)
                    hp_params['optimizer_path'] = optimizer_path + 'objects{}'.format(n_objects)
                    hp_params['random_state'] = random_state
                    hp_params['learning_problem'] = learning_problem
                    hp_params['validation_loss'] = lp_metric_dict[learning_problem].get(validation_loss, None)

                    time_taken = duration_till_now(start)
                    logger.info("Time Taken till now: {}  milliseconds".format(seconds_to_time(time_taken)))
                    time_eout_eval = get_duration_seconds('5H')
                    logger.info(
                        "Time spared for the out of sample evaluation : {} ".format(seconds_to_time(time_eout_eval)))

                    total_duration = duration - time_taken - time_eout_eval
                    hp_fit_params['n_iter'] = 20
                    hp_fit_params['total_duration'] = total_duration
                    hp_fit_params['cv_iter'] = inner_cv
                    optimizer_model = ParameterOptimizer(**hp_params)
                    optimizer_model.fit(X_train, Y_train, **hp_fit_params)
                    batch_size = X_test.shape[0]
                    s_pred, y_pred = get_scores(optimizer_model, batch_size)
                    metric_loss = categorical_accuracy_np(Y_test, y_pred)
                    logger.info(ERROR_OUTPUT_STRING.format("CategoricalAccuracy", str(np.mean(metric_loss)), n_objects))
                    eval_results[n_objects] = metric_loss
                rows_list.append(eval_results)
                save_results(rows_list, df_path)
