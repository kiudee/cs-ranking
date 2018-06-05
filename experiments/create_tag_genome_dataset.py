import inspect
import logging
import os

from csrank import TagGenomeObjectRankingDatasetReader
from csrank.util import configure_logging_numpy_keras

if __name__ == '__main__':
    DIR_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    log_path = os.path.join(DIR_PATH, 'logs', 'prepare_tag_genome.log')
    configure_logging_numpy_keras(log_path=log_path)
    logger = logging.getLogger('TagGenome')
    logger.info("ccs req id : {}".format(os.environ['CCS_REQID']))
    tg = TagGenomeObjectRankingDatasetReader()
