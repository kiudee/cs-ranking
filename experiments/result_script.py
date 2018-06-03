import inspect
import logging
import os

import pandas as pd
from tabulate import tabulate

from csrank.util import configure_logging_numpy_keras, print_dictionary
from experiments.dbconnection import DBConnector

if __name__ == '__main__':
    DIR_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    log_path = os.path.join(DIR_PATH, 'logs', 'results.log')
    configure_logging_numpy_keras(log_path=log_path)
    logger = logging.getLogger('Result parsing')
    config_file_path = os.path.join(DIR_PATH, 'config', 'clusterdb.json')
    self = DBConnector(config_file_path=config_file_path, is_gpu=True, schema='master')
    self.init_connection()
    results_table = "results.object_ranking"
    avail_jobs = "{}.avail_jobs".format(self.schema)
    select_st = "SELECT dataset_params, dataset, learner, kendallstau, spearmancorrelation, zerooneaccuracy, ndcgtopall, zerooneranklossties  from {0} INNER JOIN {1} ON {0}.job_id = {1}.job_id where {1}.dataset='letor_or'".format(
        results_table, avail_jobs)
    self.cursor_db.execute(select_st)
    data_2007 = []
    data_2008 = []
    for job in self.cursor_db.fetchall():
        values = list(job.values())
        keys = list(job.keys())
        columns = keys[2:]
        vals = values[2:] + [1.0 - values[-1]]
        if job['dataset_params']['year'] == 2007:
            values_ = ['Letor2007'] + vals
            data_2007.append(values_)
        if job['dataset_params']['year'] == 2008:
            values_ = ['Letor2008'] + vals
            data_2008.append(values_)
    cols = ['Dataset'] + columns + ['zeroonerankaccuracy']
    df = pd.DataFrame(data_2008, columns=cols)
    df.to_csv('2008.csv')
    df = pd.DataFrame(data_2007, columns=cols)
    df.to_csv('2007.csv')
    print(tabulate(df))
