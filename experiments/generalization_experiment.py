"""Experiment runner for one dataset.

Usage:
  generalization_experiment.py (--dataset=<dataset> --dataset_type=<dataset_type> --learning_problem=<learning_problem>)

  generalization_experiment.py (-h | --help)

Arguments:
    FILE       An argument for passing in a file.
Options:
  -h --help             Show this screen.
  --dataset=<dataset>         The dataset name
  --dataset_type=<dataset_type>   The dataset variant
  --learning_problem=<learning_problem>   The Learning Problem variant
"""
import copy
import hashlib
import inspect
import logging
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from docopt import docopt
from pymc3.variational.callbacks import CheckParametersConvergence
from sklearn.model_selection import ShuffleSplit
from skopt.utils import load

from csrank.constants import OBJECT_RANKING, CHOICE_FUNCTION, DISCRETE_CHOICE
from csrank.experiments import *
from csrank.experiments.util import learners
from csrank.metrics_np import categorical_accuracy_np, zero_one_rank_loss_for_scores_np, f1_measure
from csrank.tensorflow_util import configure_numpy_keras
from csrank.util import print_dictionary, setup_logging

max_objects = 30
N_OBJECTS_ARRAY = np.arange(3, max_objects)
OBJECTS = "Objects"
MODEL = "aModel"

ERROR_OUTPUT_STRING = 'Out of sample error {} : {} for n_objects {}'
LOGS_FOLDER = 'logs'
OPTIMIZER_FOLDER = 'optimizers'
PREDICTIONS_FOLDER = 'predictions'
MODEL_FOLDER = 'models'
RESULT_FOLDER = 'results'

DIR_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

metrics_dict = {OBJECT_RANKING: zero_one_rank_loss_for_scores_np, DISCRETE_CHOICE: categorical_accuracy_np,
                CHOICE_FUNCTION: f1_measure}
metric_name_dict = {OBJECT_RANKING: 'ZeroOneRankLoss', DISCRETE_CHOICE: 'CategoricalAccuracy',
                    CHOICE_FUNCTION: 'F1Score'}
LEARNERS_DICTIONARY = {DISCRETE_CHOICE: DCFS, CHOICE_FUNCTION: CHOICE_FUNCTIONS, OBJECT_RANKING: OBJECT_RANKERS}


def get_hash_string(logger, job):
    keys = ['learner', 'dataset_params', 'learner_params', 'hp_ranges', 'dataset']
    hash_string = ""
    for k in keys:
        hash_string = hash_string + str(k) + ':' + str(job[k])
    hash_object = hashlib.sha1(hash_string.encode())
    hex_dig = hash_object.hexdigest()
    logger.info("Job_id {} Hash_string {}".format(job.get('job_id', None), str(hex_dig)))
    return str(hex_dig)[:4]


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
    dataFrame.sort_index(axis=1, inplace=True)
    dataFrame.to_csv(df_path, index=OBJECTS)


if __name__ == '__main__':
    arguments = docopt(__doc__)
    start = datetime.now()
    dataset = str(arguments['--dataset'])
    dataset_type = str(arguments['--dataset_type'])
    learning_problem = str(arguments['--learning_problem'])
    log_path = os.path.join(DIR_PATH, LOGS_FOLDER, "generalization_{}_{}.log".format(dataset_type, learning_problem))
    df_path = os.path.join(DIR_PATH, RESULT_FOLDER, "generalization_{}_{}.csv".format(dataset_type, learning_problem))
    config_file_path = os.path.join(DIR_PATH, 'config', 'clusterdb.json')

    setup_logging(log_path=log_path)
    logger = logging.getLogger('Generalization')
    dbConnector = DBConnector(config_file_path=config_file_path)
    dbConnector.init_connection()
    select_st = "SELECT * FROM {0} WHERE {0}.dataset=\'{1}\' AND dataset_params->>'dataset_type'=\'{2}\' " \
                "AND {0}.fold_id={3}".format('{0}', dataset, dataset_type, 0)
    logger.info("DB config filePath {}".format(config_file_path))
    logger.info("Arguments {}".format(arguments))
    configure_numpy_keras(seed=42)
    models_done = []
    df = None
    if os.path.isfile(df_path):
        df = pd.read_csv(df_path, index_col=0)
        models_done = list(df.columns)
    logger.info("Models done {}".format(models_done))
    run_jobs = []
    if learning_problem == DISCRETE_CHOICE:
        dbConnector.cursor_db.execute(select_st.format('discrete_choice.avail_jobs'))
        for job in dbConnector.cursor_db.fetchall():
            run_jobs.append(dict(job))
        logger.info("Query {}".format(select_st.format('discrete_choice.avail_jobs')))
        dbConnector.cursor_db.execute(select_st.format('pymc3_discrete_choice.avail_jobs'))
        for job in dbConnector.cursor_db.fetchall():
            run_jobs.append(dict(job))
        logger.info("Query {}".format(select_st.format('pymc3_discrete_choice.avail_jobs')))
    elif learning_problem == OBJECT_RANKING:
        dbConnector.cursor_db.execute(select_st.format('object_ranking.avail_jobs'))
        for job in dbConnector.cursor_db.fetchall():
            run_jobs.append(dict(job))
        logger.info("Query {}".format(select_st.format('object_ranking.avail_jobs')))
    elif learning_problem == CHOICE_FUNCTION:
        dbConnector.cursor_db.execute(select_st.format('choice_functions.avail_jobs'))
        for job in dbConnector.cursor_db.fetchall():
            run_jobs.append(dict(job))
        logger.info("Query {}".format(select_st.format('choice_functions.avail_jobs')))
    dbConnector.close_connection()
    all_learners = {}
    for job in run_jobs:
        name, param, dataset = job["learner"], job["learner_params"], job["dataset"]
        param['n_objects'], param['n_object_features'] = job['dataset_params']['n_objects'], job['dataset_params'].get(
            'n_features', 0)
        hash_file = os.path.join(DIR_PATH, MODEL_FOLDER, "{}.h5".format(job['hash_value']))
        if param['n_object_features'] == 0:
            if 'tag_genome' in dataset:
                param['n_object_features'] = 1128
            elif 'mnist' in dataset:
                param['n_object_features'] = 128
        learner = learners[name](**param)
        learner.hash_file = hash_file
        all_learners[name] = learner
        logger.info('Learner {}, params {}, for dataset {}'.format(name, param, dataset))
    if dataset_type == 'median':
        N_OBJECTS_ARRAY = np.arange(3, max_objects, step=2)
    logger.info("N_OBJECTS_ARRAY {}".format(N_OBJECTS_ARRAY))
    for job_description in run_jobs:
        rows_list = []
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
        logger.info("################## {} ##################".format(learner_name.title()))
        jd = {k: job_description[k] for k in ('learner', 'dataset_params', 'learner_params', 'hp_ranges', 'dataset',
                                              'hash_value', 'fit_params') if k in job_description}
        logger.info(print_dictionary(jd))
        dataset_params['random_state'] = random_state
        dataset_params['fold_id'] = fold_id
        dataset_reader = get_dataset_reader(dataset_name, dataset_params)
        inner_cv = ShuffleSplit(n_splits=n_inner_folds, test_size=0.1, random_state=random_state)
        if learner_name in [MNL, PCL, NLM, GEV, MLM]:
            fit_params['random_seed'] = seed + fold_id
            fit_params['vi_params']["callbacks"] = [
                CheckParametersConvergence(diff="absolute", tolerance=0.01, every=50)]
        optimizer = None
        try:
            optimizer = load(optimizer_path)
            if "ps" in optimizer.acq_func:
                best_i = np.argmin(np.array(optimizer.yi)[:, 0])
            else:
                best_i = np.argmin(optimizer.yi)
            best_point = optimizer.Xi[best_i]
            best_loss = optimizer.yi[best_i]
            logger.info("Best parameters so far with a loss of {:.4f}:\n {}".format(best_loss, best_point))
        except KeyError as e:
            print("I/O error({0})".format(e))
        except ValueError:
            print("Could not convert data to an integer.")
        except:
            print("Unexpected error:", sys.exc_info()[0])
        if optimizer is None:
            logger.info("Learner Skipped {}".format(learner_name))
        add_in_name = ''
        if job_description['learner_params'].get("add_zeroth_order_model", False):
            add_in_name = '_zero'
        model_name = '{}{}'.format(learner_name, add_in_name)
        X_train, Y_train, X_test, Y_test = dataset_reader.get_single_train_test_split()
        metric_function = metrics_dict[learning_problem]
        metric_name = metric_name_dict[learning_problem]
        if model_name in models_done + [PCL]:
            logger.info("Model already present {}".format(model_name))
        if optimizer is None:
            logger.info("Optimizer is not loaded properly {}".format(model_name))
        if learner_name in [FATELINEAR_RANKER, FETALINEAR_RANKER, FATELINEAR_DC, FETALINEAR_DC] or \
                (model_name not in models_done and learner_name not in [PCL]):
            if learner_name in [FATELINEAR_RANKER, FETALINEAR_RANKER, FATELINEAR_DC, FETALINEAR_DC]:
                fit_params["epochs"] = 500
                model_name = learner_name + "_2"
            learner = all_learners[learner_name]
            logger.info("learner params {}".format(print_dictionary(learner_params)))
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
            if optimizer is not None:
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
                    learner.set_tunable_parameters(**best_learner_params)
            learner.fit(X_train, Y_train, **fit_params)
            eval_results = {MODEL: model_name}
            for n_objects in N_OBJECTS_ARRAY:
                if "synthetic" in dataset:
                    dataset_reader.kwargs['n_objects'] = n_objects
                else:
                    dataset_reader.n_objects = n_objects
                X_train, Y_train, X_test, Y_test = dataset_reader.get_single_train_test_split()
                batch_size = X_test.shape[0]
                if "clear_memory" in dir(learner):
                    learner.clear_memory(n_objects=n_objects)
                s_pred, y_pred = get_scores(learner, batch_size, X_test, Y_test, logger)
                if metric_function in metrics_on_predictions:
                    metric_loss = metric_function(Y_test, y_pred)
                else:
                    metric_loss = metric_function(Y_test, s_pred)
                logger.info("Learned on {} objects and predicting on others ".format(dataset_params['n_objects']))
                logger.info(ERROR_OUTPUT_STRING.format(metric_name, str(np.mean(metric_loss)), n_objects))
                eval_results[n_objects] = metric_loss

            logger.info("Saving  model {}".format(model_name))
            rows_list.append(eval_results)
            save_results(rows_list, df_path)
    if "224" in str(os.environ.get('HOSTNAME', 0)):
        f = open("{}/.hash_value".format(os.environ['HOME']), "w+")
        f.write("generalization_{}_{}".format(dataset_type, learning_problem) + "\n")
        f.close()
