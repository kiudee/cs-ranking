import inspect
import logging
import os

import pandas as pd
import tabulate

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
    results_table = "results.object_ranking"
    avail_jobs = "{}.avail_jobs".format(self.schema)
    select_st = "SELECT dataset_params, dataset, kendallstau, spearmancorrelation, zerooneranklossties, zerooneaccuracy, ndcgtopall  from {0} INNER JOIN {1} ON {0}.job_id = {1}.job_id where {1}.learner='listnet'".format(
        results_table, avail_jobs)
    self.cursor_db.execute(select_st)
    data_m = []
    data_h = []
    data_d = []
    for job in self.cursor_db.fetchall():
        values = list(job.values())
        keys = list(job.keys())
        columns = keys[2:]
        if job['dataset_params']['dataset_type'] == 'medoid':
            data_m.append(values[2:])
        if job['dataset_params']['dataset_type'] == 'hypervolume':
            data_h.append(values[2:])
        if job['dataset_params']['dataset_type'] == 'basic':
            data_d.append(values[2:])
    cols = ['Dataset'] + columns + ['zeroonerankaccuracy']
    data_com = []
    for name, data in zip(['Medoid', 'HyperVolume', 'DepthBasic'], [data_m, data_h, data_d]):
        one_row = [name]
        df = pd.DataFrame(data, columns=columns)
        df['zeroonerankaccuracy'] = 1.0 - df["zerooneranklossties"]
        std = df.std(axis=0).values
        mean = df.mean(axis=0).values
        one_row.extend(["{:.3f}+-{:.3f}".format(m, s) for m, s in zip(mean, std)])
        data_com.append(one_row)
    df = pd.DataFrame(data_com, columns=cols)
    df.to_csv('complete.csv')
    print(tabulate(df))
