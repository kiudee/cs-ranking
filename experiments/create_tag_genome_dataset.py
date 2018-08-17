import inspect
import logging
import os

from csrank import TagGenomeObjectRankingDatasetReader
from csrank.tensorflow_util import configure_numpy_keras
from csrank.util import setup_logging

if __name__ == '__main__':
    DIR_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    log_path = os.path.join(DIR_PATH, 'logs', 'prepare_tag_genome.log')
    setup_logging(log_path=log_path)
    configure_numpy_keras(seed=42)
    logger = logging.getLogger('TagGenome')
    logger.info("ccs req id : {}".format(os.environ['CCS_REQID']))
    tg = TagGenomeObjectRankingDatasetReader()
