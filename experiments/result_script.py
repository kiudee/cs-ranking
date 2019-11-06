import hashlib
import inspect
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from csrank.constants import CHOICE_FUNCTION, DISCRETE_CHOICE, OBJECT_RANKING
from csrank.experiments import DBConnector, lp_metric_dict
from csrank.experiments.constants import OR_MODELS, DCMS, CHOICE_MODELS

sns.set(color_codes=True)
plt.style.use('default')

__all__ = ['learners_map', 'get_dataset_name', 'get_results_for_dataset', 'get_combined_results', 'MNIST', 'LETOR',
           'get_combined_results_plot', 'create_df', 'metric_name_dict', 'learning_model_dict', 'bar_plot_for_problem']
DIR_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
config_file_path = os.path.join(DIR_PATH, 'config', 'clusterdb.json')
learners_map = {CHOICE_FUNCTION: "ChoiceModel", OBJECT_RANKING: "Ranker", DISCRETE_CHOICE: "DiscreteChoiceModel"}
metric_name_dict = {OBJECT_RANKING: "$0/1$ Ranking Accuracy", DISCRETE_CHOICE: 'Accuracy',
                    CHOICE_FUNCTION: '$F_1$' + 'Measure'}
learning_model_dict = {OBJECT_RANKING: OR_MODELS, DISCRETE_CHOICE: DCMS, CHOICE_FUNCTION: CHOICE_MODELS}
m_map = {"categoricalaccuracy": "Accuracy", 'hammingaccuracy': 'HammingAccuracy', 'aucscore': 'AUC-Score',
         'kendallstau': 'KendallsTau', 'spearmancorrelation': "SpearmanCorrelation", "f1score": '$F_1$' + 'Measure',
         "zeroonerankaccurancy": "$0/1$ Ranking Accuracy", "expectedreciprocalrank": "ExpectedReciprocalRank",
         "zerooneaccuracy": "$0/1$ Accuracy", 'averageprecisionscore': 'AveragePrecisionScore',
         "ndcgtopall": "NDCGTopAll"}
fig_param = {'facecolor': 'w', 'edgecolor': 'w', 'transparent': False, 'dpi': 800, 'bbox_inches': 'tight',
             'pad_inches': 0.05}
MNIST = "MNIST"
LETOR = "LETOR"



def get_hash_string(logger, job):
    keys = ['learner', 'dataset_params', 'learner_params', 'hp_ranges', 'dataset']
    hash_string = ""
    for k in keys:
        hash_string = hash_string + str(k) + ':' + str(job[k])
    hash_object = hashlib.sha1(hash_string.encode())
    hex_dig = hash_object.hexdigest()
    # logger.info("Job_id {} Hash_string {}".format(job.get('job_id', None), str(hex_dig)))
    return str(hex_dig)[:4]


def get_letor_string(dp, lp):
    y = dp.get('year', None)
    n = str(dp.get("n_objects", 5))
    if y == None:
        ext = "{}_n_{}".format("EXPEDIA", n)
    else:
        ext = "y_{}_n_{}".format(y, n)
        if lp in [OBJECT_RANKING, DISCRETE_CHOICE]:
            ext = 'list_' + ext
    return ext


def get_dataset_name(name):
    named = dict()
    named["NEAREST_NEIGHBOUR_MEDOID".title()] = "Nearest Neighbour"
    named["NEAREST_NEIGHBOUR".title()] = "Tag Genome Similar"
    named["DISSIMILAR_NEAREST_NEIGHBOUR".title()] = "Tag Genome Dissimilar"
    named["CRITIQUE_FIT_LESS".title()] = "Best Critique-Fit Movie d=-1"
    named["CRITIQUE_FIT_MORE".title()] = "Best Critique-Fit Movie d=+1"
    named["DISSIMILAR_CRITIQUE_LESS".title()] = "Impostor Critique-Fit Movie d=-1"
    named["DISSIMILAR_CRITIQUE_MORE".title()] = "Impostor Critique-Fit Movie d=+1"
    named["UNIQUE_MAX_OCCURRING".title()] = "{}-Mode".format(MNIST)
    named["MODE".title()] = "{}-Mode".format(MNIST)
    named["UNIQUE".title()] = "{}-Unique".format(MNIST)
    named["SUSHI_DC".title()] = "Sushi"
    named["Y_2007_N_10"] = "{}-MQ2007 10 Objects".format(LETOR)
    named["Y_2007_N_5"] = "{}-MQ2007 5 Objects".format(LETOR)
    named["Y_2008_N_10"] = "{}-MQ2008 10 Objects".format(LETOR)
    named["Y_2008_N_5"] = "{}-MQ2008 5 Objects".format(LETOR)
    named["list_Y_2007_N_10".title()] = "{}-MQ2007list 10 Objects".format(LETOR)
    named["list_Y_2007_N_5".title()] = "{}-MQ2007list 5 Objects".format(LETOR)
    named["list_Y_2008_N_10".title()] = "{}-MQ2008list 10 Objects".format(LETOR)
    named["list_Y_2008_N_5".title()] = "{}-MQ2008list 5 Objects".format(LETOR)
    named["EXPEDIA_N_10".title()] = "Expedia 10 Objects"
    named["EXPEDIA_N_5".title()] = "Expedia 5 Objects"
    named["pareto".title()] = "Pareto-front"
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
            job['dataset'] = get_letor_string(job['dataset_params'], learning_problem)
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
                job['dataset'] = get_letor_string(job['dataset_params'], learning_problem)
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
        df_full.rename(columns={'subset01loss': 'subset $0/1$ accuracy', 'hammingloss': 'hammingaccuracy'},
                       inplace=True)
    if learning_problem == OBJECT_RANKING:
        df_full['zeroonerankloss'] = 1 - df_full['zeroonerankloss']
        df_full.rename(
            columns={'zeroonerankloss': 'zeroonerankaccurancy', 'zerooneranklossties': 'expectedreciprocalrank'},
            inplace=True)

    columns = list(df_full.columns)
    return df_full, columns


def create_df(columns, data, learning_problem):
    for i in range(len(columns)):
        if "top" in columns[i] and 'ndcg' not in columns[i]:
            columns[i] = "Top-{}".format(columns[i].split("topk")[-1])
        elif "se" in columns[i] and columns[i] not in ['dataset', 'subset $0/1$ accuracy']:
            c = columns[i].split("se")[0]
            columns[i] = m_map.get(c, c.title()) + "Se"
        else:
            columns[i] = m_map.get(columns[i], columns[i].title())
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
            std = np.around(group.std(axis=0, skipna=True).values, 4)
            mean = np.around(group.mean(axis=0, skipna=True).values, 4)
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
        columns.append("{}se".format(c))
    columns = cols + columns
    for dataset, dgroup in df_full.groupby(['dataset']):
        for learner, group in dgroup.groupby(['learner']):
            one_row = [dataset, learner]
            std = np.around(group.std(axis=0, skipna=True).values, 4)
            mean = np.around(group.mean(axis=0, skipna=True).values, 4)
            if np.all(np.isnan(std)):
                one_row.extend([m for m in mean])
                one_row.extend([0.0 for m in mean])
            else:
                std_err = [s for s in std]
                one_row.extend([m for m in mean])
                one_row.extend([se for se in std_err])
            data.append(one_row)
    return create_df(columns, data, learning_problem)


def bar_plot_for_problem(df, learning_problem, start, params, extension):
    bar_width = 0.20
    opacity = 0.6
    learning_model = learners_map[learning_problem]
    col = metric_name_dict[learning_problem]
    FOLDER = "journalresults"
    fname = os.path.join(DIR_PATH, FOLDER, "{}_{}.{}")
    colse = col + 'Se'
    u_models = [m for m in learning_model_dict[learning_problem] if m in df[learning_model].unique()]
    u_datasets = np.array(df.Dataset.unique())
    if learning_problem == DISCRETE_CHOICE:
        mid = int(len(u_datasets) / 2)
        df1 = df[df.Dataset.str.contains('|'.join(u_datasets[0:mid]))]
        df2 = df[df.Dataset.str.contains('|'.join(u_datasets[mid:]))]
        dfs = [df1, df2]
    else:
        dfs = [df]
    # for sub_df, sub_plot in zip([df1, df2], [ax1, ax2]):
    fig_param['format'] = extension
    for i, sub_df in enumerate(dfs):
        fig, ax = plt.subplots(figsize=(8, 5), frameon=True, edgecolor='k', facecolor='white')
        uds = np.array(sub_df.Dataset.unique())
        init_index = bar_width * (len(u_models) + 1) * np.arange(1, len(uds) + 1)
        index = init_index
        init_index = init_index + bar_width * (len(u_models)) / 2.0
        end = 1.01
        for model in u_models:
            acc = sub_df[sub_df[learning_model] == model][col].values
            errors = sub_df[sub_df[learning_model] == model][colse].values
            plt.bar(x=index, height=acc, yerr=errors, width=bar_width, alpha=opacity, label=model)
            index = index + bar_width
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.legend(**params)
        plt.ylim(start, end)
        plt.xticks(init_index, uds)
        plt.yticks(np.arange(start, end, 0.05))
        plt.xlabel("Dataset")
        plt.ylabel(col)
        plt.tight_layout()
        f_path = fname.format(learning_problem, i, extension)
        fig_param['fname'] = f_path
        plt.savefig(**fig_param)
        plt.show()
        i += 1
