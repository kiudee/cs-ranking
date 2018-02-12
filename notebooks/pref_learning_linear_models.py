import inspect
import os
import warnings

from matplotlib.colors import LinearSegmentedColormap
from sklearn.linear_model import LinearRegression

from csrank.objectranking.expected_rank_regression import ExpectedRankRegression

warnings.simplefilter('ignore')
import pandas as pd
from csrank.deprecated.trip_advisor_dataset_reader import TripAdviosrDataset
from csrank.objectranking.rank_svm import *
from csrank.util import kendalls_mean, create_dir_recursively
from csrank.graph_util import generate_plots, heat_map
from scipy.stats import rankdata
import seaborn as sns
from matplotlib import pyplot as plt
from colorspacious import cspace_convert as conv
import numpy as np
import logging


def actual_ranks_regression(X, orderings):
    X1 = []
    Y_single = []

    for features, rank in zip(X, orderings):
        X1.extend(features)
        Y_single.extend(rank)
    X1 = np.array(X1)
    Y_single = np.array(Y_single)
    return X1, Y_single


def norm_function(weights):
    # weights = (weights - np.mean(weights)) / np.std(weights)
    weights = np.abs(weights) / np.sum(np.abs(weights))
    return weights


def run_for_one_config():
    for i, test_city in enumerate(city_names):
        train_cities = city_names[:i] + city_names[i + 1:]
        X, Y, _, headers = tp_dataset_reader.data_for_cities(train_cities, features_to_remove=features_to_remove)
        err_model = ExpectedRankRegression(n_features=len(headers), alpha=0.0, fit_intercept=fit_intercept,
                                           random_state=random_state)
        ranksvm = RankSVM(n_features=len(headers), fit_intercept=fit_intercept, random_state=random_state)
        lin_model = LinearRegression(fit_intercept=fit_intercept)

        headers, features, actual_ranks, normalized_ranks = tp_dataset_reader.data_for_city(test_city,
                                                                                            features_to_remove=features_to_remove)
        ranksvm.fit(X, Y)
        err_model.fit(X, Y)

        # ERR model
        predicted_ranks = err_model.predict(features)
        tau = kendalls_mean(predicted_ranks, actual_ranks)
        kendalls_err[test_city] = tau
        logger.info('*******************************************************')
        logger.info("TestCity: {} ".format(test_city))
        logger.info('Expected Rank Regression: ' + repr(tau))
        weights_err_dict[test_city] = norm_function(err_model.weights)

        # RankSVM model
        predicted_ranks = ranksvm.predict(features)
        tau = kendalls_mean(predicted_ranks, actual_ranks)
        kendalls_ranksvm[test_city] = tau
        logger.info('RankSVM: ' + repr(tau))
        weights_ranksvm_dict[test_city] = norm_function(ranksvm.weights)

        # Linear Regression on actual ranks
        X, Y = actual_ranks_regression(X, Y)
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)
        lin_model.fit(X, Y)
        features[0] = scaler.fit_transform(features[0])
        predicted_ranks = rankdata(lin_model.predict(features[0])).astype(int)[None, :]
        tau = kendalls_mean(predicted_ranks, actual_ranks)
        kendalls_linear_reg[test_city] = tau
        logger.info('Regression on actual ranks: ' + repr(tau))

        # Baseline model
        _, data_test, _, _ = tp_dataset_reader.data_for_city(test_city, features_to_remove=baseline_features)
        predicted_scores = [-h for h in data_test]
        predicted_ranks = rankdata(predicted_scores).astype(int)[None, :]
        tau = kendalls_mean(predicted_ranks, actual_ranks)
        kendalls_baseline[test_city] = tau
        logger.info('Baseline: ' + repr(tau))
        logger.info('*******************************************************')

    dataFrame = pd.DataFrame([kendalls_err, kendalls_ranksvm, kendalls_linear_reg, kendalls_baseline]).T
    dataFrame.columns = ['Expected Rank Regression', 'RankSVM', "RegressionActualRanks", "Baseline"]
    file_path = os.path.join(dirname, output_folder, output_file + ".csv")
    std = list(dataFrame.std())
    dataFrame.loc["mean"] = list(dataFrame.mean())
    dataFrame.loc["std"] = std
    dataFrame.to_csv(file_path)
    file_path = os.path.join(dirname, output_folder, output_file + ".png")
    generate_plots(weights_err_dict, weights_ranksvm_dict, city_names, "LIN-REG", "SVM", headers, file_path,
                   col1=blue, col2=green, fit_intercept=fit_intercept)


if __name__ == '__main__':

    plt.style.use("seaborn-dark-palette")
    sns.set_style("dark")
    sns.set_color_codes("bright")
    colors = sns.color_palette("Set1", n_colors=5)
    sns.set_palette(colors)
    blue = colors[1]
    green = colors[2]
    dirname = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

    random_state = np.random.RandomState(seed=42)
    tp_dataset_reader = TripAdviosrDataset()
    city_names = tp_dataset_reader.city_names
    kendalls_err = {}
    kendalls_ranksvm = {}
    kendalls_baseline = {}
    kendalls_linear_reg = {}
    weights_err_dict = {}
    weights_ranksvm_dict = {}
    baseline_features = list(tp_dataset_reader.all_headers)
    baseline_features.remove('rtg_overall_computed')
    output_folder = "new_outputs"
    file_path = os.path.join(dirname, output_folder)
    create_dir_recursively(file_path)
    logging.basicConfig(filename=os.path.join(dirname, output_folder, 'logs.log'),
                        level=logging.DEBUG,
                        format='%(asctime)s %(levelname)-8s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger(name='Experiment')
    default_features_removed = ["rtg_overall", 'rtg_overall_computed']

    for i in range(4):
        if i == 0:
            fit_intercept = False
            features_to_remove = default_features_removed
        if i == 1:
            fit_intercept = True
            features_to_remove = ['eco_award', 'travelers_choice', 'certificate_excellence', 'n_families', 'n_couples',
                                  'n_singles', 'n_business', "rtg_overall", 'rtg_overall_computed', 'n_rating']
        if i == 2:
            fit_intercept = False
            features_to_remove = ['eco_award', 'travelers_choice', 'certificate_excellence', 'n_families', 'n_couples',
                                  'n_singles', 'n_business', "rtg_overall", 'rtg_overall_computed', 'n_rating']
        if i == 3:
            fit_intercept = True
            features_to_remove = default_features_removed

        if (not fit_intercept):
            output_file = "result_no_intercept"
        else:
            output_file = "result"
        if (len(features_to_remove) > len(default_features_removed)):
            output_file = output_file + "_features_removed"

        #run_for_one_config()

    dataframe = tp_dataset_reader.datasets_dictionaries[city_names[0]]
    # plot_features_distributions(dataframe,
    #                             mypath=os.path.join(dirname, output_folder, "normed"), color=blue,
    #                             preprocessing=True)
    # plot_features_distributions(dataframe,
    #                             mypath=os.path.join(dirname, output_folder, "unnormed"), color=blue)

    X, Y, norm_ranks, headers = tp_dataset_reader.data_for_cities(city_names, features_to_remove=["rtg_overall"])
    X, Y = actual_ranks_regression(X, Y)

    # Y = []
    # for nr in norm_ranks:
    #     Y.extend(nr)
    # Y = np.array(Y)
    Y = Y[:, None]

    X = np.hstack((X, Y))
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    headers.append("Popindex")
    file_path = os.path.join(dirname, output_folder, "population_covariance_popindex.png")

    blue_ucs = conv(blue, 'sRGB1', 'CAM02-UCS')
    white_ucs = conv((1., 1., 1.), 'sRGB1', 'CAM02-UCS')

    pal_mat = np.clip(
        conv((np.linspace(0, 1, num=10)[:, None] * (blue_ucs - white_ucs) + white_ucs), 'CAM02-UCS', 'sRGB1'),
        0, 1
    )

    cmap = LinearSegmentedColormap.from_list(
        'test', pal_mat, N=100)
    heat_map(file_path, X, headers, cmap="vlag")
