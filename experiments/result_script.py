import hashlib
import inspect
import os

import numpy as np
import pandas as pd

from csrank.constants import CHOICE_FUNCTION, DISCRETE_CHOICE, OBJECT_RANKING
from csrank.experiments import DBConnector, lp_metric_dict

__all__ = ['learners_map', 'get_dataset_name', 'get_results_for_dataset', 'get_combined_results',
           'get_combined_results_plot', 'create_df']
DIR_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
config_file_path = os.path.join(DIR_PATH, 'config', 'clusterdb.json')
learners_map = {CHOICE_FUNCTION: "ChoiceModel", OBJECT_RANKING: "Ranker", DISCRETE_CHOICE: "DiscreteChoiceModel"}


def get_hash_string(logger, job):
    keys = ['learner', 'dataset_params', 'learner_params', 'hp_ranges', 'dataset']
    hash_string = ""
    for k in keys:
        hash_string = hash_string + str(k) + ':' + str(job[k])
    hash_object = hashlib.sha1(hash_string.encode())
    hex_dig = hash_object.hexdigest()
    logger.info("Job_id {} Hash_string {}".format(job.get('job_id', None), str(hex_dig)))
    return str(hex_dig)[:4]


def get_letor_string(dp):
    y = dp.get('year', None)
    n = str(dp.get("n_objects", 5))
    if y == None:
        ext = "{}_n_{}".format("EXPEDIA", n)
    else:
        ext = "y_{}_n_{}".format(y, n)
    return ext


def get_dataset_name(name):
    named = dict()
    named["NEAREST_NEIGHBOUR_MEDOID".title()] = "Nearest Neighbour"
    named["NEAREST_NEIGHBOUR".title()] = "Most Similar Movie"
    named["DISSIMILAR_NEAREST_NEIGHBOUR".title()] = "Most Dissimilar Movie"
    named["CRITIQUE_FIT_LESS".title()] = "Best Critique-Fit Movie d=-1"
    named["CRITIQUE_FIT_MORE".title()] = "Best Critique-Fit Movie d=+1"
    named["DISSIMILAR_CRITIQUE_LESS".title()] = "Impostor Critique-Fit Movie d=-1"
    named["DISSIMILAR_CRITIQUE_MORE".title()] = "Impostor Critique-Fit Movie d=+1"
    named["UNIQUE_MAX_OCCURRING".title()] = "Mode"
    named["HYPERVOLUME".title()] = "Pareto"
    named["SUSHI_DC".title()] = "Sushi"
    named["Y_2007_N_10"] = "MQ2007 10 Objects"
    named["Y_2007_N_5"] = "MQ2007 5 Objects"
    named["Y_2008_N_10"] = "MQ2008 10 Objects"
    named["Y_2008_N_5"] = "MQ2008 5 Objects"
    named["EXPEDIA_N_10".title()] = "Expedia 10 Objects"
    named["EXPEDIA_N_5".title()] = "Expedia 5 Objects"
    if name not in named.keys():
        named[name] = name.lower().title()
    return named[name]


def get_results_for_dataset(DATASET, logger, learning_problem=DISCRETE_CHOICE, del_jid=True):
    results_table = 'results.{}'.format(learning_problem)
    if learning_problem == CHOICE_FUNCTION:
        schema = learning_problem + 's'
    else:
        schema = learning_problem
    keys = list(lp_metric_dict[learning_problem].keys())
    metrics = ', '.join([x for x in keys])
    start = 3
    select_jobs = "SELECT learner_params, dataset_params, hp_ranges, {0}.job_id, dataset, learner, {3} from {0} " \
                  "INNER JOIN {1} ON {0}.job_id = {1}.job_id where {1}.dataset=\'{2}\'"

    self = DBConnector(config_file_path=config_file_path, is_gpu=False, schema=schema)
    self.init_connection()
    avail_jobs = "{}.avail_jobs".format(schema)
    select_st = select_jobs.format(results_table, avail_jobs, DATASET, metrics)
    self.cursor_db.execute(select_st)
    data = []
    for job in self.cursor_db.fetchall():
        job = dict(job)
        ext = get_hash_string(logger=logger, job=job)
        if job['learner_params'].get("add_zeroth_order_model", False):
            job['learner'] = job['learner'] + '_zero'
        job['learner'] = job['learner'] + '_' + ext
        if "letor" in job['dataset'] or "exp" in job['dataset']:
            job['dataset'] = get_letor_string(job['dataset_params'])
        elif "sushi" in job['dataset']:
            job['dataset'] = job['dataset']
        else:
            job['dataset'] = job['dataset_params']['dataset_type']
        job['dataset'] = job['dataset'].title()
        values = list(job.values())
        keys = list(job.keys())
        columns = keys[start:]
        vals = values[start:]
        data.append(vals)
    self.close_connection()
    if learning_problem == DISCRETE_CHOICE:
        self.init_connection()
        avail_jobs = "{}.avail_jobs".format('pymc3_' + DISCRETE_CHOICE)
        select_st = select_jobs.format(results_table, avail_jobs, DATASET, metrics)
        # print(select_st)
        self.cursor_db.execute(select_st)
        for job in self.cursor_db.fetchall():
            job = dict(job)
            ext = get_hash_string(logger=logger, job=job)
            job['learner'] = job['learner'] + '_' + ext
            if job['dataset'] in ["letor_dc", "exp_dc"]:
                job['dataset'] = get_letor_string(job['dataset_params'])
            elif "sushi" in job['dataset']:
                job['dataset'] = job['dataset']
            else:
                job['dataset'] = job['dataset_params']['dataset_type']
            job['dataset'] = job['dataset'].title()
            values = list(job.values())
            keys = list(job.keys())
            columns = keys[start:]
            vals = values[start:]
            data.append(vals)
    df_full = pd.DataFrame(data, columns=columns)
    df_full = df_full.sort_values('dataset')
    if del_jid:
        del df_full['job_id']
    if learning_problem == CHOICE_FUNCTION:
        df_full['subset01loss'] = 1 - df_full['subset01loss']
        df_full['hammingloss'] = 1 - df_full['hammingloss']
        df_full.rename(columns={'subset01loss': 'subset01accuracy', 'hammingloss': 'hammingaccuracy'}, inplace=True)
    columns = list(df_full.columns)

    return df_full, columns


def create_df(columns, data, learning_problem):
    for i in range(len(columns)):
        if "categorical" in columns[i]:
            if "accuracy" in columns[i]:
                columns[i] = "Categorical" + columns[i].split("categorical")[-1].title()
            else:
                columns[i] = "Top-{}".format(columns[i].split("topk")[-1])
        else:
            columns[i] = columns[i].title()
            if columns[i] == 'Learner':
                columns[i] = learners_map[learning_problem]
    df = pd.DataFrame(data, columns=columns)
    df.sort_values(by='Dataset')
    if learning_problem == DISCRETE_CHOICE:
        del df[columns[7]]
        if "se" in columns[-1].lower():
            del df[columns[-1]]
    return df


def get_combined_results(DATASET, logger, learning_problem, latex_row=False):
    df_full, columns = get_results_for_dataset(DATASET, logger, learning_problem, True)
    data = []
    df_full = df_full.replace(np.inf, 0)
    for dataset, dgroup in df_full.groupby(['dataset']):
        for learner, group in dgroup.groupby(['learner']):
            one_row = [dataset, learner]
            std = np.around(group.std(axis=0, skipna=True).values, 3)
            mean = np.around(group.mean(axis=0, skipna=True).values, 3)
            if np.all(np.isnan(std)):
                one_row.extend(["{:.4f}".format(m) for m in mean])
            else:
                std_err = [s / np.sqrt(len(group)) for s in std]
                if latex_row:
                    one_row.extend(["{:.3f}({:.0f})".format(m, s * 1e3) for m, s in zip(mean, std)])
                else:
                    one_row.extend(["{:.3f}±{:.3f}".format(m, s) for m, s in zip(mean, std)])
            data.append(one_row)
    return create_df(columns, data, learning_problem)


def get_combined_results_plot(DATASET, logger, learning_problem, latex_row=False):
    df_full, cols = get_results_for_dataset(DATASET, logger, learning_problem, True)
    df_full = df_full.replace(np.inf, 0)
    data = []
    columns = []
    idx = cols.index('learner') + 1
    for c in cols[idx:]:
        if 'categorical' in c:
            columns.append("{}se".format(c))
    columns = cols + columns
    for dataset, dgroup in df_full.groupby(['dataset']):
        for learner, group in dgroup.groupby(['learner']):
            one_row = [dataset, learner]
            std = np.around(group.std(axis=0, skipna=True).values, 3)
            mean = np.around(group.mean(axis=0, skipna=True).values, 3)
            if np.all(np.isnan(std)):
                one_row.extend([m for m in mean])
                one_row.extend([0.0 for m in mean])
            else:
                std_err = [s for s in std]
                one_row.extend([m for m in mean])
                one_row.extend([se for se in std_err])
            data.append(one_row)
    return create_df(columns, data, learning_problem)
