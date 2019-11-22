import hashlib
import inspect
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from csrank.constants import CHOICE_FUNCTION, DISCRETE_CHOICE, OBJECT_RANKING
from csrank.experiments.constants import OR_MODELS, DCMS, CHOICE_MODELS, OBJECT_RANKERS, DCFS, CHOICE_FUNCTIONS, PCL, \
    RANDOM_DC, RANDOM_RANKER, RANDOM_CHOICE, MNL, GLM_CHOICE, GEV, NLM, MLM, RANKSVM_CHOICE, RANKSVM_DC
from csrank.experiments.dbconnection import DBConnector
from csrank.experiments.util import lp_metric_dict

sns.set(color_codes=True)
plt.style.use('default')

__all__ = ['learners_map', 'get_dataset_name', 'get_results_for_dataset', 'get_combined_results', 'MNIST', 'LETOR',
           'get_combined_results_plot', 'create_df', 'metric_name_dict', 'learning_models_dict', 'bar_plot_for_problem',
           'get_ranges_dataset', 'learning_functions_dict', 'create_final_result']
DIR_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
config_file_path = os.path.join(DIR_PATH, 'config', 'clusterdb.json')
learners_map = {CHOICE_FUNCTION: "ChoiceModel", OBJECT_RANKING: "Ranker", DISCRETE_CHOICE: "DiscreteChoiceModel"}
metric_name_dict = {OBJECT_RANKING: "$0/1$ Ranking Accuracy", DISCRETE_CHOICE: 'Accuracy',
                    CHOICE_FUNCTION: '$F_1$-measure'}
learning_models_dict = {OBJECT_RANKING: OR_MODELS, DISCRETE_CHOICE: DCMS, CHOICE_FUNCTION: CHOICE_MODELS}
learning_functions_dict = {OBJECT_RANKING: OBJECT_RANKERS, DISCRETE_CHOICE: DCFS, CHOICE_FUNCTION: CHOICE_FUNCTIONS}

m_map = {"categoricalaccuracy": "Accuracy", 'hammingaccuracy': 'HammingAccuracy', 'aucscore': 'AUC-Score',
         'kendallstau': 'KendallsTau', 'spearmancorrelation': "SpearmanCorrelation", "f1score": '$F_1$-measure',
         "zeroonerankaccurancy": "$0/1$ Ranking Accuracy", "expectedreciprocalrank": "ExpectedReciprocalRank",
         "zerooneaccuracy": "$0/1$ Accuracy", 'averageprecisionscore': 'AveragePrecisionScore',
         "ndcgtopall": "NDCGTopAll"}
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
    named["NEAREST_NEIGHBOUR".title()] = "Tag Genome Similar Movie"
    named["DISSIMILAR_NEAREST_NEIGHBOUR".title()] = "Tag Genome Dissimilar Movie"
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


def get_ranges_dataset(DATASET, logger, learning_problem=DISCRETE_CHOICE, latex_row=False):
    results_table = 'results.{}'.format(learning_problem)
    if learning_problem == CHOICE_FUNCTION:
        schema = learning_problem + 's'
    else:
        schema = learning_problem
    keys = list(lp_metric_dict[learning_problem].keys())
    metric = keys[0]
    start = 2
    select_jobs = "SELECT learner_params, dataset_params, dataset, learner, hp_ranges, {3} from {0} " \
                  "INNER JOIN {1} ON {0}.job_id = {1}.job_id where {1}.dataset=\'{2}\'"

    self = DBConnector(config_file_path=config_file_path, is_gpu=False, schema=schema)
    self.init_connection()
    avail_jobs = "{}.avail_jobs".format(schema)
    select_st = select_jobs.format(results_table, avail_jobs, DATASET, metric)
    self.cursor_db.execute(select_st)
    data = []
    jobs = self.cursor_db.fetchall()
    for job in jobs:
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
        select_st = select_jobs.format(results_table, avail_jobs, DATASET, metric)
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
    columns = list(df_full.columns)
    data = []
    df_full = df_full.replace(np.inf, 0)
    for dataset, dgroup in df_full.groupby(['dataset']):
        for learner, group in dgroup.groupby(['learner']):
            one_row = [dataset, learner, group['hp_ranges'].values[0]]

            def flatten(dicta):
                d = {}
                for key, val in sorted(dicta.items(), key=lambda item: item[0]):
                    for k, v in sorted(val.items(), key=lambda item: item[0]):
                        if k not in ['regularization', 'loss_function']:
                            d[k] = v
                for key, val in sorted(d.items(), key=lambda item: item[0]):
                    d[key] = val
                return d

            # print(one_row[-1], flatten(one_row[-1]))
            one_row[-1] = flatten(one_row[-1])
            del group['hp_ranges']
            std = np.around(group.std(axis=0, skipna=True).values, 4)[0]
            mean = np.around(group.mean(axis=0, skipna=True).values, 4)[0]
            one_row.append("{:.3f}±{:.3f}".format(mean, std))
            data.append(one_row)
    df = create_df(columns, data, learning_problem)
    return df


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
    df.sort_values(by=['Dataset', learners_map[learning_problem]])
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


def get_val(val):
    vals = [float(x) for x in re.findall(r"[-+]?\d*\.\d+|\d+", val)]
    if len(vals) == 1:
        x = [vals[0], vals[0] - 0.0]
    else:
        x = [vals[0], vals[0] - vals[1]]
    return x


def create_final_result(dataset, logger, learning_problem, dataset_function=get_combined_results, latex_row=False):
    df_full = dataset_function(dataset, logger, learning_problem, latex_row=latex_row)
    data = []
    functions = learning_functions_dict[learning_problem]
    learning_model = learners_map[learning_problem]
    models_dict = dict(zip(functions, learning_models_dict[learning_problem]))
    for dataset, df in df_full.groupby(['Dataset']):
        for m in functions:
            row = df[df[learning_model].str.contains(m)].values
            onerow = None
            if len(row) > 1:
                if dataset_function == get_combined_results:
                    values = np.array([get_val(val[2]) for val in row])
                elif dataset_function == get_ranges_dataset:
                    values = np.array([get_val(val[3]) for val in row])
                else:
                    values = np.array([[val[2], val[2] - val[7]] for val in row])
                maxi = np.where(values[:, 0] == values[:, 0][np.argmax(values[:, 0])])[0][0]
                logger.error("dataset {} model {}, vals {}, maxi {}".format(dataset, row[:, 1], values, maxi))
                row = row[maxi]
                row[1] = models_dict[m]
                onerow = row
            elif len(row) == 1:
                row[0][1] = models_dict[m]
                onerow = row[0]
            if onerow is not None:
                onerow[0] = get_dataset_name(onerow[0])
                data.append(onerow)
    columns = df_full.columns
    dataFrame = pd.DataFrame(data, columns=columns)
    if dataset_function == get_combined_results or dataset_function == get_combined_results_plot:
        dataFrame = dataFrame.sort_values(by=[columns[0], columns[2]], ascending=[True, False])
    elif dataset_function == get_ranges_dataset:
        del dataFrame[columns[-1]]
        searchFor = ['Nearest Neighbour', "5 Objects", "Critique", 'Largest', 'Median']
        dataFrame = dataFrame[~dataFrame['Dataset'].str.contains('|'.join(searchFor))]
        searchFor = [models_dict.get(PCL, 'None'), models_dict.get(RANDOM_DC, 'None'),
                     models_dict.get(RANDOM_RANKER, 'None'), models_dict.get(RANDOM_CHOICE, 'None'),
                     models_dict.get(MNL, 'None'), models_dict.get(GLM_CHOICE, 'None'),
                     models_dict.get(RANKSVM_CHOICE, 'None'), models_dict.get(GEV, 'None'),
                     models_dict.get(RANKSVM_DC, 'None'), models_dict.get(NLM, 'None'), models_dict.get(MLM, 'None')]
        dataFrame = dataFrame[~dataFrame[learning_model].str.contains('|'.join(searchFor))]
        # dataFrame = dataFrame[~dataFrame['Hp_Ranges'].str.contains('{}')]
        dataFrame.replace(to_replace=r' 10 Objects', value='', regex=True, inplace=True)
        dataFrame.rename(columns={learners_map[learning_problem]: "LearningModel"}, inplace=True)

    return dataFrame


def bar_plot_for_problem(df, learning_problem, start, params, extension):
    fig_param = {'facecolor': 'w', 'edgecolor': 'w', 'transparent': False, 'dpi': 800, 'bbox_inches': 'tight',
                 'pad_inches': 0.05}
    bar_width = 0.20
    opacity = 0.6
    learning_model = learners_map[learning_problem]
    col = metric_name_dict[learning_problem]
    FOLDER = "journalresults"
    fname = os.path.join(DIR_PATH, FOLDER, "{}_{}.{}")
    colse = col + 'Se'
    u_models = [m for m in learning_models_dict[learning_problem] if m in df[learning_model].unique()]
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
        fig, ax = plt.subplots(figsize=(5.63, 2.6), frameon=True, edgecolor='k', facecolor='white')
        uds = np.array(sub_df.Dataset.unique())
        init_index = bar_width * (len(u_models) + 1) * np.arange(1, len(uds) + 1)
        index = init_index
        init_index = init_index + bar_width * (len(u_models)) / 2.0
        end = 1.01
        handels = []
        labels = []
        for model in u_models:
            acc = sub_df[sub_df[learning_model] == model][col].values
            errors = sub_df[sub_df[learning_model] == model][colse].values
            handle = ax.bar(x=index, height=acc, yerr=errors, width=bar_width, alpha=opacity, label=model)
            index = index + bar_width
            handels.append(handle)
            labels.append(model)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.legend(**params)
        plt.ylim(start, end)
        plt.xticks(init_index, uds, fontsize=7)
        plt.yticks(np.arange(start, end, 0.1), fontsize=7)
        #plt.xlabel("Dataset", fontsize=8)
        col = col.replace("$0/1$ Ranking Accuracy", "$0/1$ Ranking \n Accuracy")
        plt.ylabel(col, fontsize=8)
        plt.tight_layout()
        f_path = fname.format(learning_problem, i, extension)
        fig_param['fname'] = f_path
        #fig_param[' bbox_extra_artists'] = (legd,)
        plt.savefig(**fig_param)
        plt.show()
        i += 1


def bar_plot_for_problem2(df, learning_problem, start, params, extension):
    fig_param = {'facecolor': 'w', 'edgecolor': 'w', 'transparent': False, 'dpi': 800, 'bbox_inches': 'tight',
                 'pad_inches': 0.05}
    bar_width = 0.20
    opacity = 0.6
    learning_model = learners_map[learning_problem]
    col = metric_name_dict[learning_problem]
    FOLDER = "journalresults"
    fname = os.path.join(DIR_PATH, FOLDER, "{}.{}")
    colse = col + 'Se'
    u_models = [m for m in learning_models_dict[learning_problem] if m in df[learning_model].unique()]
    u_datasets = np.array(df.Dataset.unique())
    fig_param['format'] = extension

    if learning_problem == DISCRETE_CHOICE:
        mid = int(len(u_datasets) / 2)
        df1 = df[df.Dataset.str.contains('|'.join(u_datasets[0:mid]))]
        df2 = df[df.Dataset.str.contains('|'.join(u_datasets[mid:]))]
        dfs = [df1, df2]
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5.63, 5.3), frameon=True, edgecolor='k', facecolor='white')
        subps = [ax1, ax2]
    else:
        dfs = [df]
        fig, ax = plt.subplots(figsize=(5.63, 2.6), frameon=True, edgecolor='k', facecolor='white')
        subps = [ax]
    for sub_df, ax in zip(dfs, subps):
        uds = np.array(sub_df.Dataset.unique())
        init_index = bar_width * (len(u_models) + 1) * np.arange(1, len(uds) + 1)
        index = init_index
        init_index = init_index + bar_width * (len(u_models)) / 2.0
        end = 1.01
        handels = []
        labels = []
        for model in u_models:
            acc = sub_df[sub_df[learning_model] == model][col].values
            errors = sub_df[sub_df[learning_model] == model][colse].values
            handle = ax.bar(x=index, height=acc, yerr=errors, width=bar_width, alpha=opacity, label=model)
            index = index + bar_width
            handels.append(handle)
            labels.append(model)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(labelsize=7)
        ax.set_ylim(start, end)
        ax.set_yticks(np.arange(start, end, 0.1))
        ax.set_xticks(init_index)
        ax.set_xticklabels(uds)
        col = col.replace("$0/1$ Ranking Accuracy", "$0/1$ Ranking \n Accuracy")
        ax.set_ylabel(col, fontsize=8)

    plt.legend(**params)
    plt.tight_layout()
    f_path = fname.format(learning_problem, extension)
    fig_param['fname'] = f_path
    plt.savefig(**fig_param)
    plt.show()
