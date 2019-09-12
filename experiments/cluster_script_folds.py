"""Thin experiment runner which takes all simulation parameters from a database.

Usage:
  experiment_cv.py --cindex=<id> --config_fileName=<config_fileName> --isgpu=<bool> --schema=<schema>
  experiment_cv.py (-h | --help)

Arguments:
  FILE                  An argument for passing in a file.

Options:
  -h --help                             Show this screen.
  --cindex=<cindex>                     Index given by the cluster to specify which job
                                        is to be executed [default: 0]
  --config_fileName=<config_fileName>   File name of the database config
  --isgpu=<bool>                        Boolean to show if the gpu is to be used or not
  --schema=<schema>                     Schema containing the job information
"""
import copy
import inspect
import logging
import os
import pickle as pk
import sys
import traceback
from datetime import datetime

import h5py
import numpy as np
from docopt import docopt
from sklearn.model_selection import ShuffleSplit

from csrank.experiments import *
from csrank.metrics import make_ndcg_at_k_loss
from csrank.metrics_np import topk_categorical_accuracy_np
from csrank.tensorflow_util import configure_numpy_keras, get_mean_loss
from csrank.tuning import ParameterOptimizer
from csrank.util import create_dir_recursively, duration_till_now, seconds_to_time, \
    print_dictionary, get_duration_seconds, setup_logging

DIR_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
LOGS_FOLDER = 'logs'
OPTIMIZER_FOLDER = 'optimizers'
PREDICTIONS_FOLDER = 'predictions'
MODEL_FOLDER = 'models'
ERROR_OUTPUT_STRING = 'Out of sample error %s : %0.4f'

if __name__ == "__main__":
    start = datetime.now()

    ######################## DOCOPT ARGUMENTS: #################################
    arguments = docopt(__doc__)
    cluster_id = int(arguments["--cindex"])
    config_fileName = arguments["--config_fileName"]
    is_gpu = bool(int(arguments["--isgpu"]))
    schema = arguments["--schema"]
    ###################### POSTGRESQL PARAMETERS ###############################
    config_file_path = os.path.join(DIR_PATH, 'config', config_fileName)
    dbConnector = DBConnector(config_file_path=config_file_path, is_gpu=is_gpu, schema=schema)
    if 'CCS_REQID' in os.environ.keys():
        cluster_id = int(os.environ['CCS_REQID'])
    dbConnector.fetch_job_arguments(cluster_id=cluster_id)
    baseline = False
    if dbConnector.job_description is not None:
        try:
            job_description = copy.deepcopy(dict(dbConnector.job_description))
            seed = int(job_description["seed"])
            job_id = int(job_description["job_id"])
            fold_id = int(job_description["fold_id"])
            dataset_name = job_description["dataset"]
            n_inner_folds = int(job_description["inner_folds"])
            dataset_params = job_description["dataset_params"]
            learner_name = job_description["learner"]
            fit_params = job_description["fit_params"]
            learner_params = job_description["learner_params"]
            duration = job_description["duration"]
            hp_iters = int(job_description["hp_iters"])
            hp_ranges = job_description["hp_ranges"]
            hp_fit_params = job_description["hp_fit_params"]
            learning_problem = job_description["learning_problem"]
            experiment_schema = job_description["experiment_schema"]
            experiment_table = job_description["experiment_table"]
            validation_loss = job_description["validation_loss"]
            hash_value = job_description["hash_value"]
            random_state = np.random.RandomState(seed=seed + fold_id)
            log_path = os.path.join(DIR_PATH, LOGS_FOLDER, "{}.log".format(hash_value))
            optimizer_path = os.path.join(DIR_PATH, OPTIMIZER_FOLDER, "{}".format(hash_value))
            create_dir_recursively(log_path, True)
            create_dir_recursively(optimizer_path, True)
            setup_logging(log_path=log_path)
            configure_numpy_keras(seed=seed)
            logger = logging.getLogger('Experiment')
            logger.info("DB config filePath {}".format(config_file_path))
            logger.info("Arguments {}".format(arguments))
            logger.info("Job Description {}".format(print_dictionary(job_description)))
            duration = get_duration_seconds(duration)

            dataset_params['random_state'] = random_state
            dataset_params['fold_id'] = fold_id
            dataset_reader = get_dataset_reader(dataset_name, dataset_params)
            X_train, Y_train, X_test, Y_test = dataset_reader.get_single_train_test_split()
            n_objects = log_test_train_data(X_train, X_test, logger)
            del dataset_reader

            inner_cv = ShuffleSplit(n_splits=n_inner_folds, test_size=0.1, random_state=random_state)
            hash_file = os.path.join(DIR_PATH, MODEL_FOLDER, "{}.h5".format(hash_value))
            create_dir_recursively(hash_file, True)
            learner_params['n_objects'], learner_params['n_object_features'] = X_train.shape[1:]
            logger.info("learner params {}".format(print_dictionary(learner_params)))
            hp_params = create_optimizer_parameters(fit_params, hp_ranges, learner_params, learner_name, hash_file)
            hp_params['optimizer_path'] = optimizer_path
            hp_params['random_state'] = random_state
            hp_params['learning_problem'] = learning_problem
            hp_params['validation_loss'] = lp_metric_dict[learning_problem].get(validation_loss, None)

            time_taken = duration_till_now(start)
            logger.info("Time Taken till now: {}  milliseconds".format(seconds_to_time(time_taken)))
            time_eout_eval = get_duration_seconds('1H')
            logger.info("Time spared for the out of sample evaluation : {} ".format(seconds_to_time(time_eout_eval)))

            total_duration = duration - time_taken - time_eout_eval
            hp_fit_params['n_iter'] = hp_iters
            hp_fit_params['total_duration'] = total_duration
            hp_fit_params['cv_iter'] = inner_cv
            optimizer_model = ParameterOptimizer(**hp_params)
            optimizer_model.fit(X_train, Y_train, **hp_fit_params)
            hp_fit_params['n_iter'] = 0
            for fold_id in range(5):
                if fold_id != 0:
                    del X_train, X_test, Y_test, Y_train
                    random_state = np.random.RandomState(seed=seed + fold_id)
                    dataset_params['random_state'] = random_state
                    dataset_params['fold_id'] = fold_id
                    dataset_reader = get_dataset_reader(dataset_name, dataset_params)
                    X_train, Y_train, X_test, Y_test = dataset_reader.get_single_train_test_split()
                    current_job_id = dbConnector.clone_job(cluster_id=cluster_id, fold_id=fold_id)
                    del optimizer_model
                    optimizer_model = ParameterOptimizer(**hp_params)
                    optimizer_model.fit(X_train, Y_train, **hp_fit_params)
                else:
                    current_job_id = job_id
                logger.info('current job id {}'.format(current_job_id))
                if isinstance(X_test, dict):
                    batch_size = 10000
                else:
                    size = sys.getsizeof(X_test)
                    batch_size = X_test.shape[0]
                    logger.info("Test dataset size {}".format(size))
                s_pred, y_pred = get_scores(optimizer_model.model, batch_size, X_test, Y_test, logger)
                if fold_id == 0:
                    if isinstance(s_pred, dict):
                        pred_file = os.path.join(DIR_PATH, PREDICTIONS_FOLDER, "{}.pkl".format(hash_value))
                        create_dir_recursively(pred_file, True)
                        f = open(pred_file, "wb")
                        pk.dump(s_pred, f)
                        f.close()
                    else:
                        pred_file = os.path.join(DIR_PATH, PREDICTIONS_FOLDER, "{}.h5".format(hash_value))
                        create_dir_recursively(pred_file, True)
                        f = h5py.File(pred_file, 'w')
                        f.create_dataset('scores', data=s_pred)
                        f.close()
                    logger.info("Saved predictions at: {}".format(pred_file))

                results = {'job_id': str(current_job_id), 'cluster_id': str(cluster_id)}
                for name, evaluation_metric in lp_metric_dict[learning_problem].items():
                    predictions = s_pred
                    if evaluation_metric in metrics_on_predictions:
                        logger.info("Metric on predictions")
                        predictions = y_pred
                    if "NDCG" in name:
                        evaluation_metric = make_ndcg_at_k_loss(k=n_objects)
                        predictions = y_pred
                    if "CategoricalTopK{}" == name:
                        k = int(n_objects / 2) + 1
                        evaluation_metric = topk_categorical_accuracy_np(k=k)
                        name = name.format(k)
                    metric_loss = get_mean_loss(evaluation_metric, Y_test, predictions)
                    logger.info(ERROR_OUTPUT_STRING % (name, metric_loss))
                    if np.isnan(metric_loss):
                        results[name] = "\'Infinity\'"
                    else:
                        results[name] = "{0:.4f}".format(metric_loss)
                dbConnector.insert_results(experiment_schema=experiment_schema, experiment_table=experiment_table,
                                           results=results)
                if fold_id != 0:
                    dbConnector.mark_running_job_finished(current_job_id)
            dbConnector.mark_running_job_finished(job_id)
        except Exception as e:
            if hasattr(e, 'message'):
                message = e.message
            else:
                message = e
            logger.error(traceback.format_exc())
            message = "exception{}".format(str(message))
            dbConnector.append_error_string_in_running_job(job_id=job_id, error_message=message)
        except:
            logger.error(traceback.format_exc())
            message = "exception{}".format(sys.exc_info()[0].__name__)
            dbConnector.append_error_string_in_running_job(job_id=job_id, error_message=message)
        finally:
            if "224" in str(cluster_id):
                f = open("{}/.hash_value".format(os.environ['HOME']), "w+")
                f.write(hash_value + "\n")
                f.close()
