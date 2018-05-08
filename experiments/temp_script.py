import inspect
import logging
import os

from csrank.util import configure_logging_numpy_keras

if __name__ == '__main__':
    print("in temp script")
    DIR_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    log_path = os.path.join(DIR_PATH, 'logs', 'test.log')
    configure_logging_numpy_keras(log_path=log_path)
    logger = logging.getLogger('Generalization Experiment')
    logger.info("ccs req id : {}".format(os.environ['CCS_REQID']))
