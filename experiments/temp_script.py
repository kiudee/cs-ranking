import inspect
import logging
import os

from csrank.util import configure_logging_numpy_keras

if __name__ == '__main__':
    LOGS_FOLDER = 'logs'
    DIR_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    configure_logging_numpy_keras(log_path=os.path.join(DIR_PATH, LOGS_FOLDER, "test.log"))
    logger = logging.getLogger('Generalization Experiment')
    logger.info("ccs req id : {}".format(os.environ['CCS_REQID']))