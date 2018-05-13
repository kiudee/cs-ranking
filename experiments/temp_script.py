import inspect
import logging
import os

from csrank.util import configure_logging_numpy_keras
from experiments.dbconnection import DBConnector

if __name__ == '__main__':
    LOGS_FOLDER = 'logs'
    OPTIMIZER_FOLDER = 'optimizers'
    DIR_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    log_path = os.path.join(DIR_PATH, 'logs', 'test.log')
    configure_logging_numpy_keras(log_path=log_path)
    logger = logging.getLogger('Generalization Experiment')
    logger.info("ccs req id : {}".format(os.environ['CCS_REQID']))
    config_file_path = os.path.join(DIR_PATH, 'config', 'clusterdb.json')
    dbConnector = DBConnector(config_file_path=config_file_path, is_gpu=True, schema='master')
    dbConnector.rename_all_jobs(DIR_PATH, LOGS_FOLDER, OPTIMIZER_FOLDER)
