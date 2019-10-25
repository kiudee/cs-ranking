"""Experiment runner for one dataset.

Usage:
  performance_set_size.py (--dataset=<dataset> --dataset_type=<dataset_type> --model=<model> --run_per=<run_per>)

  performance_set_size.py (-h | --help)

Arguments:
    FILE       An argument for passing in a file.
Options:
  -h --help             Show this screen.
  --dataset=<dataset>         The dataset name
  --dataset_type=<dataset_type>   The dataset variant
  --model=<model>   The model variant
  --run_per=<run_per> Boolean to run the performance results
"""
import copy
import inspect
import logging
import os
from datetime import datetime

import numpy as np
import pandas as pd
import pymc3 as pm
from docopt import docopt
from sklearn.model_selection import ShuffleSplit
from skopt.utils import load

from csrank.constants import OBJECT_RANKING, CHOICE_FUNCTION, DISCRETE_CHOICE
from csrank.experiments import *
from csrank.experiments.util import learners
from csrank.metrics_np import categorical_accuracy_np, zero_one_rank_loss_for_scores_np, f1_measure
from csrank.tensorflow_util import configure_numpy_keras
from csrank.util import print_dictionary, get_duration_seconds, setup_logging

OPTIMIZE_ON_OBJECTS = [5, 7, 15, 17]

N_OBJECTS_ARRAY = np.arange(3, 20)
OBJECTS = "Objects"
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
    if not os.path.isfile(df_path):
        dataFrame = df
        df.insert(0, OBJECTS, N_OBJECTS_ARRAY)
    else:
        dataFrame = pd.read_csv(df_path, index_col=0)
        for col in list(df.columns):
            dataFrame[col] = np.array(df[col])
    dataFrame.to_csv(df_path, index=OBJECTS)


if __name__ == '__main__':
    arguments = docopt(__doc__)
    start = datetime.now()
    dataset = str(arguments['--dataset'])
    dataset_type = str(arguments['--dataset_type'])
    model = str(arguments['--model'])
    run_per = bool(int(arguments['--run_per']))
    run_opt = False
    log_path = os.path.join(DIR_PATH, LOGS_FOLDER, "performance_sets_{}_{}_{}.log".format(dataset, dataset_type, model))
    df_path = os.path.join(DIR_PATH, RESULT_FOLDER,
                           "performance_sets_{}_{}_{}.csv".format(dataset, dataset_type, model))
    config_file_path = os.path.join(DIR_PATH, 'config', 'clusterdb.json')
    metrics_dict = {OBJECT_RANKING: zero_one_rank_loss_for_scores_np, DISCRETE_CHOICE: categorical_accuracy_np,
                    CHOICE_FUNCTION: f1_measure}
    metric_name_dict = {OBJECT_RANKING: 'ZeroOneRankLoss', DISCRETE_CHOICE: 'CategoricalAccuracy',
                        CHOICE_FUNCTION: 'F1Score'}
    setup_logging(log_path=log_path)
    logger = logging.getLogger('PerformanceSetSizes')
    dbConnector = DBConnector(config_file_path=config_file_path)
    dbConnector.init_connection()
    select_st = "SELECT * FROM {0} WHERE {0}.dataset=\'{1}\' AND dataset_params->>'dataset_type'=\'{2}\' " \
                "AND {0}.learner=\'{3}\' AND {0}.fold_id={4}".format('{0}', dataset, dataset_type, model, 0)
    models_done = []
    df = None
    if os.path.isfile(df_path):
        df = pd.read_csv(df_path, index_col=0)
        models_done = list(df.columns)
    logger.info("Models done {}".format(models_done))
    models_done = []
    run_jobs = []
    dbConnector.cursor_db.execute(select_st.format('discrete_choice.avail_jobs'))
    for job in dbConnector.cursor_db.fetchall():
        run_jobs.append(dict(job))
    dbConnector.cursor_db.execute(select_st.format('pymc3_discrete_choice.avail_jobs'))
    for job in dbConnector.cursor_db.fetchall():
        run_jobs.append(dict(job))
    dbConnector.cursor_db.execute(select_st.format('choice_functions.avail_jobs'))
    for job in dbConnector.cursor_db.fetchall():
        run_jobs.append(dict(job))
    dbConnector.cursor_db.execute(select_st.format('discrete_choice.avail_jobs'))
    for job in dbConnector.cursor_db.fetchall():
        run_jobs.append(dict(job))
    dbConnector.close_connection()
    for job in run_jobs:
        logger.info("learner {} learner_params {}".format(job["learner"], job["learner_params"]))
    job_description = run_jobs[0]
    rows_list = []
    duration = get_duration_seconds('7D')
    logger.info("DB config filePath {}".format(config_file_path))
    logger.info("Arguments {}".format(arguments))
    logger.info("Run optimization {}".format(run_opt))
    if dataset_type == 'median':
        N_OBJECTS_ARRAY = np.arange(3, 20, step=2)
    logger.info("N_OBJECTS_ARRAY {}".format(N_OBJECTS_ARRAY))
    logger.info("OPTIMIZE_ON_OBJECTS {}".format(OPTIMIZE_ON_OBJECTS))
    logger.info("Model {} Dataset {}".format(model, dataset))

    seed = int(job_description["seed"])
    job_id = int(job_description["job_id"])
    fold_id = int(job_description["fold_id"])
    dataset_name = job_description["dataset"]
    n_inner_folds = int(job_description["inner_folds"])
    dataset_params = job_description["dataset_params"]
    learner_name = job_description["learner"]
    fit_params = copy.deepcopy(job_description["fit_params"])
    learner_params = job_description["learner_params"]
    hp_iters = int(job_description["hp_iters"])
    hp_ranges = copy.deepcopy(job_description["hp_ranges"])
    hp_fit_params = copy.deepcopy(job_description["hp_fit_params"])
    learning_problem = job_description["learning_problem"]
    experiment_schema = job_description["experiment_schema"]
    experiment_table = job_description["experiment_table"]
    validation_loss = job_description["validation_loss"]
    hash_value = job_description["hash_value"]
    random_state = np.random.RandomState(seed=seed + fold_id)
    optimizer_path = os.path.join(DIR_PATH, OPTIMIZER_FOLDER, "{}".format(hash_value))
    hash_file = os.path.join(DIR_PATH, MODEL_FOLDER, "{}.h5".format(hash_value))

    configure_numpy_keras(seed=seed)
    dataset_params['random_state'] = random_state
    dataset_params['fold_id'] = fold_id
    dataset_reader = get_dataset_reader(dataset_name, dataset_params)
    inner_cv = ShuffleSplit(n_splits=n_inner_folds, test_size=0.1, random_state=random_state)
    if learner_name in [MNL, PCL, NLM, GEV, MLM]:
        fit_params['random_seed'] = seed + fold_id
        fit_params['vi_params']["callbacks"] = [pm.callbacks.CheckParametersConvergence(diff="absolute", tolerance=0.01
                                                                                        , every=50)]
    optimizer = load(optimizer_path)
    if "ps" in optimizer.acq_func:
        best_i = np.argmin(np.array(optimizer.yi)[:, 0])
    else:
        best_i = np.argmin(optimizer.yi)
    best_point = optimizer.Xi[best_i]
    best_loss = optimizer.yi[best_i]

    logger.info(print_dictionary(job_description))
    logger.info("Best parameters so far with a loss of {:.4f}:\n {}".format(best_loss, best_point))
    add_in_name = ''
    if job_description['learner_params'].get("add_zeroth_order_model", False):
        add_in_name = '_zero'
    model_name = '{}{}'.format(learner_name, add_in_name)
    learner_func = learners[learner_name]
    X_train, Y_train, X_test, Y_test = dataset_reader.get_single_train_test_split()
    metric_function = metrics_dict[learning_problem]
    metric_name = metric_name_dict[learning_problem]
    df_path = os.path.join(DIR_PATH, RESULT_FOLDER, "performance_sets_{}_{}.csv".format(learning_problem, dataset_type))
    logging.info("Saving the results for dataset {} for model {}".format(dataset_type, learner_name))
    p_bool = (not (model_name in models_done and (model_name + '_gen') in models_done))
    logger.info("Present bool {}".format(p_bool))
    # Creating the learners
    learner_params['n_objects'], learner_params['n_object_features'] = X_train.shape[1:]
    all_models = {}
    for n_objects in N_OBJECTS_ARRAY:
        learner_params['n_objects'] = n_objects
        all_models[n_objects] = learner_func(**learner_params)
        if learner_name == FETA_DC:
            all_models[n_objects].max_number_of_objects = n_objects
    if not (model_name in models_done and (model_name + '_gen') in models_done):
        learner_params['n_objects'], learner_params['n_object_features'] = X_train.shape[1:]
        logger.info("learner params {}".format(print_dictionary(learner_params)))
        learner = learner_func(**learner_params)
        learner.hash_file = hash_file
        if learner_name in hp_ranges.keys():
            tuned_objects = {learner: hp_ranges.get(learner_name, {}).keys()}
        if "callbacks" in fit_params.keys():
            callbacks = []
            for key, value in fit_params.get("callbacks", {}).items():
                callback = callbacks_dictionary[key]
                callback = callback(**value)
                callbacks.append(callback)
                if key in hp_ranges.keys():
                    tuned_objects[callback] = hp_ranges[key].keys()
            fit_params["callbacks"] = callbacks
        # Setting tunable parameters
        i = 0
        for obj, parameters in tuned_objects.items():
            param_dict = dict()
            for j, p in enumerate(parameters):
                param_dict[p] = best_point[i + j]
            if isinstance(obj, learners[learner_name]):
                best_learner_params = copy.deepcopy(param_dict)
                logger.info('Best learner params {}'.format(best_learner_params))
            else:
                obj.set_tunable_parameters(**param_dict)
                logger.info('obj: {}, current parameters {}'.format(type(obj).__name__, param_dict))
            i += len(parameters)
        if learner_name != PCL:
            learner.set_tunable_parameters(**best_learner_params)
            learner.fit(X_train, Y_train, **fit_params)
        eval_results = {MODEL: model_name}
        eval_results2 = {MODEL: model_name + '_gen'}
        for n_objects in N_OBJECTS_ARRAY:
            if dataset == "synthetic_dc":
                dataset_reader.kwargs['n_objects'] = n_objects
            else:
                dataset_reader.n_objects = n_objects
            X_train, Y_train, X_test, Y_test = dataset_reader.get_single_train_test_split()
            # Fitting the learner
            learner = all_models[n_objects]
            if learner_name == FETA_DC:
                logger.info("learner params {}".format([learner._n_objects, learner.n_objects]))
            logger.info("############################ For objects {} ##############################".format(n_objects))
            log_test_train_data(X_train, X_test, logger)
            logger.info("############################## Learner 1 ####################################")
            learner.set_tunable_parameters(**best_learner_params)
            learner.fit(X_train, Y_train, **fit_params)
            batch_size = X_test.shape[0]
            s_pred, y_pred = get_scores(learner, batch_size, X_test, logger)
            metric_loss = categorical_accuracy_np(Y_test, y_pred)
            logger.info(ERROR_OUTPUT_STRING.format("CategoricalAccuracy  ", str(np.mean(metric_loss)), n_objects))
            eval_results[n_objects] = metric_loss
            if learner_name != PCL:
                logger.info("############################## Learner 2 ####################################")
                if "clear_memory" in dir(learner):
                    learner.clear_memory(n_objects=n_objects)
                batch_size = X_test.shape[0]
                s_pred, y_pred = get_scores(learner, batch_size, X_test, Y_test, logger)
                metric_loss = metric_function(Y_test, y_pred)
                logger.info("Learned on {} objects and predicting on others ".format(dataset_params['n_objects']))
                logger.info(ERROR_OUTPUT_STRING.format(metric_name, str(np.mean(metric_loss)), n_objects))
                eval_results2[n_objects] = metric_loss
        logger.info("Saving  model {}".format(model_name))
        if learner_name != PCL:
            rows_list.append(eval_results2)
            logger.info("Saving  model {}".format(model_name + '_gen'))
        save_results(rows_list, df_path)
        del learner
