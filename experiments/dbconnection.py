import hashlib
import inspect
import json
import logging
import os
from abc import ABCMeta

import psycopg2
from psycopg2.extras import DictCursor
from sklearn.utils import check_random_state

from csrank.util import print_dictionary

DIR_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))


class DBConnector(metaclass=ABCMeta):

    def __init__(self, config_filePath, is_gpu=False, random_state=None, schema='master', **kwargs):
        self.logger = logging.getLogger('DBConnector')
        self.logger.info("Path {}".format(config_filePath))
        self.random_state = check_random_state(random_state)
        self.is_gpu = is_gpu
        self.schema = schema
        self.job_description = None
        if os.path.isfile(config_filePath):
            config_file = open(config_filePath, "r")
            config = config_file.read().replace('\n', '')
            self.logger.info("Config {}".format(config))
            self.connect_params = json.loads(config)
            self.logger.info("Connection Successful")
        else:
            raise ValueError('File does not exist for the configuration of the database')

    def init_connection(self, cursor_factory=DictCursor):
        self.connection = psycopg2.connect(**self.connect_params)
        self.cursor = self.connection.cursor(cursor_factory=cursor_factory)

    def close_connection(self):
        self.connection.commit()
        self.connection.close()

    def create_hash_value(self):
        keys = ['learner', 'dataset_params', 'fit_params', 'learner_params', 'hp_ranges', 'hp_fit_params',
                'inner_folds', 'validation_loss']
        hash_string = ""
        for k, v in self.job_description.items():
            if k in keys:
                hash_string = hash_string + str(k) + ':' + str(v)
        hash_object = hashlib.sha1(hash_string.encode())
        hex_dig = hash_object.hexdigest()
        print(hash_string)
        return str(hex_dig)

    def fetch_job_arguments(self, cluster_id):
        self.init_connection()
        avail_jobs = "{}.avail_jobs".format(self.schema)
        running_jobs = "{}.running_jobs".format(self.schema)
        select_job = "SELECT job_id FROM {0} row WHERE (is_gpu = {2})AND NOT EXISTS(SELECT job_id FROM {1} r WHERE r.interrupted = FALSE AND r.job_id = row.job_id)".format(
            avail_jobs, running_jobs, self.is_gpu)

        self.cursor.execute(select_job)
        job_ids = [j for i in self.cursor.fetchall() for j in i]
        while self.job_description is None:
            try:
                run_job_id = int(self.random_state.choice(job_ids))
                self.logger.info("Job selected : {}".format(run_job_id))
                select_job = "SELECT * FROM {0} WHERE {0}.job_id = {1}".format(avail_jobs, run_job_id)
                self.cursor.execute(select_job)
                self.job_description = self.cursor.fetchone()

                hash_value = self.create_hash_value()
                self.job_description["hash_value"] = hash_value
                self.logger.info("Job_description : {}".format(print_dictionary(self.job_description)))

                update_job = "UPDATE {0} set hash_value = %(hash_value)s WHERE job_id = %(job_id)s".format(avail_jobs)
                update_dict = {'hash_value': hash_value, 'job_id': run_job_id}
                self.cursor.execute(update_job, update_dict)

                select_job = "SELECT count(*) FROM {0} WHERE {0}.job_id = {1}".format(running_jobs, run_job_id)
                self.cursor.execute(select_job)

                if self.cursor.fetchone()['count'] == 0:
                    insert_job = "INSERT INTO {0} (job_id, cluster_id ,finished, interrupted) VALUES ({1}, {2},FALSE, FALSE)".format(
                        running_jobs, run_job_id, cluster_id)
                    self.cursor.execute(insert_job)
                    if self.cursor.rowcount == 1:
                        self.logger.info("The job {} is inserted".format(run_job_id))
                else:
                    update_job = "UPDATE {0} set cluster_id = {1}, interrupted = FALSE WHERE job_id = {2}".format(
                        running_jobs, cluster_id, run_job_id)
                    self.cursor.execute(update_job)
                    if self.cursor.rowcount == 1:
                        self.logger.info("The job {} is updated".format(run_job_id))

                self.close_connection()
            except psycopg2.IntegrityError:
                self.logger.info("IntegrityError as the job {} is already assigned to another node".format(run_job_id))
                job_ids.remove(run_job_id)
            except ValueError:
                self.logger.info("ValueError as the all jobs are already assigned to another nodes")
                break

    def mark_running_job_finished(self, job_id, **kwargs):
        self.init_connection()
        running_jobs = "{}.running_jobs".format(self.schema)
        update_job = "UPDATE {0} set finished = TRUE, interrupted = FALSE where job_id = {1}".format(running_jobs,
            job_id)
        self.cursor.execute(update_job)
        if self.cursor.rowcount == 1:
            self.logger.info("The job {} is finished".format(job_id))
        self.close_connection()

    def insert_results(self, experiment_schema, experiment_table, results, **kwargs):
        self.init_connection(cursor_factory=None)
        results_table = "{}.{}".format(experiment_schema, experiment_table)
        columns = ', '.join(list(results.keys()))
        values_str = ', '.join(list(results.values()))
        insert_result = "INSERT INTO {0} ({1}) VALUES ({2})".format(results_table, columns, values_str)
        self.cursor.execute(insert_result)
        if self.cursor.rowcount == 1:
            self.logger.info("Results inserted for the job {}".format(results['job_id']))
        self.close_connection()

    def append_error_string_in_running_job(self, job_id, error_message, **kwargs):
        self.init_connection(cursor_factory=None)
        running_jobs = "{}.running_jobs".format(self.schema)
        current_message = "SELECT error_history from {0} WHERE {0}.job_id = {1}".format(running_jobs, job_id)
        self.cursor.execute(current_message)
        cur_message = self.cursor.fetchone()
        if cur_message[0] != 'NA':
            error_message = error_message + ';\n' + cur_message[0]
        update_job = "UPDATE {0} SET error_history = %s, interrupted = %s WHERE job_id = %s".format(running_jobs)
        self.cursor.execute(update_job, (error_message, True, job_id))
        if self.cursor.rowcount == 1:
            self.logger.info("The job {} is finished".format(job_id))
        self.close_connection()
