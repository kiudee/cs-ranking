import inspect
import logging
import os

import pandas as pd

from csrank.util import configure_logging_numpy_keras
from experiments.dbconnection import DBConnector

if __name__ == '__main__':
    DIR_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    log_path = os.path.join(DIR_PATH, 'logs', 'results.log')
    configure_logging_numpy_keras(log_path=log_path)
    logger = logging.getLogger('Result parsing')
    config_file_path = os.path.join(DIR_PATH, 'config', 'clusterdb.json')
    self = DBConnector(config_file_path=config_file_path, is_gpu=True, schema='master')
    self.init_connection()
    results_table = "results.object_ranking2"
    avail_jobs = "{}.avail_jobs".format(self.schema)
    select_st = "SELECT dataset_params, dataset, learner, kendallstau, spearmancorrelation, zerooneaccuracy, ndcgtopall, zerooneranklossties  from {0} INNER JOIN {1} ON {0}.job_id = {1}.job_id where {1}.dataset='letor_or'".format(
        results_table, avail_jobs)
    self.cursor_db.execute(select_st)
    data_letor = []
    data_2008 = []
    for job in self.cursor_db.fetchall():
        values = list(job.values())
        keys = list(job.keys())
        columns = keys[2:]
        vals = values[2:] + [1.0 - values[-1]]
        if job['dataset_params']['year'] == 2007:
            values_ = ['Letor2007'] + vals
            data_letor.append(values_)
        if job['dataset_params']['year'] == 2008:
            values_ = ['Letor2008'] + vals
            data_letor.append(values_)
    cols = ['Dataset'] + columns + ['zeroonerankaccuracy']
    df = pd.DataFrame(data_letor, columns=cols)
    df_path = os.path.join(DIR_PATH, 'results', 'letor.csv')
    df.to_csv(df_path)
    grouped = df.groupby(['Dataset', 'learner'])
    data = []
    for name, group in grouped:
        one_row = [name[0], str(name[1]).upper()]
        std = group.std(axis=0).values
        mean = group.mean(axis=0).values
        one_row.extend(["{:.3f}+-{:.3f}".format(m, s) for m, s in zip(mean, std)])
        data.append(one_row)
    df = pd.DataFrame(data, columns=cols)
    df_path = os.path.join(DIR_PATH, 'results', 'letor_aggregated.csv')
    df.to_csv(df_path)
