import inspect
import logging
import os
from itertools import product

import numpy as np
import pandas as pd
from skopt import load, dump

from csrank.constants import OBJECT_RANKING
from csrank.util import files_with_same_name, create_dir_recursively, rename_file_if_exist
from experiments.util import dataset_options_dict, rankers_dict, lp_metric_dict

DIR_NAME = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))


def log_best_params(file):
    opt = load(file)

    if "ps" in opt.acq_func:
        best_i = np.argmin(np.array(opt.yi)[:, 0])
        best_loss = opt.yi[best_i]
        best_params = opt.Xi[best_i]
        logger.info(
            "Best parameters so far with a loss for file {} of {:.4f} time of {:.4f}:\n {}".format(
                os.path.basename(file), best_loss[0],
                best_loss[1],
                best_params))
    else:
        best_i = np.argmin(opt.yi)
        best_loss = opt.yi[best_i]
        best_params = opt.Xi[best_i]
        logger.info(
            "Best parameters so far with a loss for file {} of {:.4f}:\n {}".format(os.path.basename(file), best_loss,
                                                                                    best_params))
    return best_loss


def remove_redundant_optimizer_models(model_path, files_list):
    logger.info('Results Files {} for Path {}'.format(files_list, os.path.basename(model_path)))
    minimum_error = 50000
    if len(files_list) >= 2:
        for file in files_list:
            try:
                opt = load(file)
                best_loss = log_best_params(file)
                if best_loss < minimum_error:
                    minimum_error = best_loss
                    if (file != model_path):
                        logger.info('Writing from the file {} to {}'.format(os.path.basename(file),
                                                                            os.path.basename(model_path)))
                        os.remove(model_path)
                        dump(opt, model_path)
            except KeyError:
                logger.error('Cannot open the file {}'.format(file))

            except ValueError:
                logger.error('Cannot open the file {}'.format(file))
    elif len(files_list) == 1:
        file = files_list[0]
        try:
            best_loss = log_best_params(file)
        except KeyError:
            logger.error('Cannot open the file {}'.format(file))
        except ValueError:
            logger.error('Cannot open the file {}'.format(file))

    if len(files_list) != 0:
        files_list.remove(model_path)
        for file in files_list:
            logger.error('Removing the File {}'.format(file))
            os.remove(file)


def remove_redundant_log_files(logs_path, logs_files_list, ranker_name, dataset):
    logger.info('Log Files {} for Path {}'.format(logs_files_list, os.path.basename(logs_path)))
    minimum_error = 50000
    if len(logs_files_list) >= 2:
        for file in logs_files_list:
            lines = np.array([line.rstrip('\n') for line in open(file)])
            out = 'zero_one_rank_loss'
            matching = [s for s in lines if out in s]
            try:
                logger.info("For File {} the error is {}".format(file, matching))
                err = float(matching[0].split(out + ' : ')[-1])
                logger.info("For File {} the zero one rank errro is {}".format(file, err))
                if err <= minimum_error:
                    minimum_error = err
                    if (file != logs_path):
                        logger.info('Renaming from the file {} to {}'.format(os.path.basename(file),
                                                                             os.path.basename(logs_path)))
                        os.remove(logs_path)
                        os.system('mv {} {}'.format(file, logs_path))
            except IndexError:
                logger.error('error {} in ranker {} is not evaluated for dataset {}'.format(out, ranker_name, dataset))
            except ValueError:
                logger.error('error {} in ranker {} is not evaluated for dataset {}'.format(out, ranker_name, dataset))


def remove_redundant_results():
    for dataset, ranker_name in product(dataset_options.keys(), ranker_options.keys()):
        model_path = os.path.join(DIR_NAME, 'optimizer_results_single_fold', '{}_{}'.format(dataset, ranker_name))
        files_list = files_with_same_name(model_path)
        remove_redundant_optimizer_models(model_path, files_list)

        logs_path = os.path.join(DIR_NAME, 'logs_single_fold', '{}_{}.log'.format(dataset, ranker_name))
        logs_files_list = files_with_same_name(logs_path)
        remove_redundant_log_files(logs_path, logs_files_list, ranker_name, dataset)


def generate_concise_results_for_dataset(dataset='medoid', directory='logs_single_fold', result_directory='results'):
    ranker_names = list(ranker_options.keys())
    ranker_names.sort()
    metric_names.sort()
    data = []
    data.append(['**************', dataset.upper(), '**************', ""])
    for ranker_name in ranker_names:
        try:
            log_path = os.path.join(DIR_NAME, directory, '{}_{}.log'.format(dataset, ranker_name))
            lines = np.array([line.rstrip('\n') for line in open(log_path)])
        except FileNotFoundError:
            logger.error('File {} is not found'.format(log_path))
            data.append(['NE' for i in range(len(metric_names))])
            continue
        one_row = []
        for out in metric_names:
            try:
                matching = [s for s in lines if out in s][0]
                if out in matching:
                    one_row.append(matching.split(out + ' : ')[-1])
            except IndexError:
                logger.error('error {} in ranker {} is not evaluated for dataset {}'.format(out, ranker_name, dataset))
                one_row.append('NE')
        data.append(one_row)
    columns = [name.upper() for name in metric_names]
    indexes = [name.upper() for name in ranker_names]
    indexes.insert(0, 'DATASET')
    dataFrame = pd.DataFrame(data, index=indexes, columns=columns)
    file_path = os.path.join(DIR_NAME, result_directory, '{}.csv'.format(dataset))
    create_dir_recursively(file_path, True)
    dataFrame.to_csv(file_path)
    return dataFrame


def create_concise_results(result_directory='results', directory='logs_single_fold'):
    df_list = []
    datasets = list(dataset_options.keys())
    datasets.sort()
    for dataset in datasets:
        dataFrame = generate_concise_results_for_dataset(dataset=dataset, directory=directory,
                                                         result_directory=result_directory)
        df_list.append(dataFrame)
    full_df = pd.concat(df_list)
    fout = os.path.join(DIR_NAME, result_directory, 'complete_results.csv')
    full_df.to_csv(fout)


def configure_logging():
    log_path = os.path.join(DIR_NAME, 'results', 'compiling_result.log')
    create_dir_recursively(log_path, True)
    log_path = rename_file_if_exist(log_path)
    global logger
    logging.basicConfig(filename=log_path, level=logging.DEBUG, format='%(asctime)s %(levelname)-8s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger(name='Compiling Results')


if __name__ == '__main__':
    configure_logging()
    dataset_options = dataset_options_dict[OBJECT_RANKING]
    ranker_options = rankers_dict[OBJECT_RANKING]
    metric_names = list(lp_metric_dict[OBJECT_RANKING].keys())
    remove_redundant_results()
    create_concise_results()
    # create_concise_results(result_directory='logs_new_experiments', directory='logs_new_experiments')
