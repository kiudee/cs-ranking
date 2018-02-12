import os

import numpy as np
import pylab
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.covariance import GraphLasso
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from csrank.util import create_dir_recursively


def generate_plots(weights_dict_1, weights_dict_2, keys,
                   label1, label2, x_values, file_path, col1="b", col2="r", fit_intercept=True):
    weights_1 = weights_dict_1[keys[0]]
    weights_2 = weights_dict_2[keys[0]]
    for key in keys:
        if (keys[0] == key):
            continue
        weights_1 = np.vstack((weights_1, weights_dict_1[key]))
        weights_2 = np.vstack((weights_2, weights_dict_2[key]))
    scaler_1 = StandardScaler().fit(weights_1)
    scaler_2 = StandardScaler().fit(weights_2)

    bar_width = 0.40
    opacity = 1.0
    series1 = scaler_1.mean_ / np.sum(np.abs(scaler_1.mean_))
    series2 = scaler_2.mean_ / np.sum(np.abs(scaler_2.mean_))
    series1_err = scaler_1.var_  # / np.sum(np.abs(scaler_1.var_))
    series2_err = scaler_2.var_  # / np.sum(np.abs(scaler_2.var_))

    if (fit_intercept):
        x_values = np.append(x_values, "intercept")
    index = np.arange(len(x_values))

    plt.grid(which="major", color='w', linestyle='-', axis='y', linewidth=0.4)
    plt.grid(which='major', color='w', linestyle='-', axis='x', linewidth=0.4)

    rects1 = plt.bar(index, series1, bar_width,
                     alpha=opacity,
                     color=col1,
                     yerr=series1_err,
                     label=label1)
    rects2 = plt.bar(index + bar_width, series2, bar_width,
                     alpha=opacity,
                     color=col2,
                     yerr=series2_err,
                     label=label2)

    plt.xlabel('Features')
    plt.ylabel('Mean and Variance')

    plt.xticks(index + bar_width / 2, x_values, rotation=90, ha='right')
    plt.legend()
    plt.tight_layout()

    figure = plt.gcf()  # get current figure
    figure.set_size_inches(14, 9)
    figure.savefig(file_path)
    plt.close(figure)
    del figure


def heat_map(file_path, X, headers, cmap=sns.color_palette("Blues")):
    model = GraphLasso()
    model.fit(X)
    Cov = model.covariance_
    std = np.diag(1. / np.sqrt(np.diag(Cov)))
    Cor = std.dot(Cov).dot(std)

    fig, ax = plt.subplots()
    # the size of A4 paper
    fig.set_size_inches(10, 8)
    ax = sns.heatmap(Cor, cmap=cmap, square=True, xticklabels=1, yticklabels=1, linewidths=.5)
    ax.set_yticklabels(headers, rotation=0, fontsize=12)
    ax.set_xticklabels(headers, rotation=90, fontsize=12)
    plt.subplots_adjust(bottom=0.4, left=0.2)

    sns.despine(left=True, bottom=True)

    plt.tight_layout()

    plt.savefig(file_path)
    plt.show()


def plot_features_distributions(dataset, mypath, color="b", preprocessing=True):
    headers = dataset.columns[0:-1]
    create_dir_recursively(mypath)
    data_frame = dataset.reset_index(drop=True)
    data_frame.id = 'index'
    if (preprocessing == True):
        for column in headers:
            column_values = np.log(np.array(data_frame[column]) + 1)
            scaler = MinMaxScaler()
            data_frame[column] = scaler.fit_transform(column_values.reshape(-1, 1)).T[0]

    for column in headers:
        x = data_frame[column]
        sns.distplot(x, kde=False, color=color)
        fig = plt.gcf()
        fig.set_size_inches(14, 9)
        file_path = os.path.join(mypath, column + "_dist.png")
        fig.savefig(file_path, dpi=600)
        plt.close(fig)
        pylab.close(fig)
        del fig

    for column in headers:
        data_frame = data_frame.sort_values(by=[column], ascending=[True])
        sns.countplot(x=str(column), data=data_frame, color=color)
        fig = plt.gcf()
        fig.set_size_inches(14, 9)
        file_path = os.path.join(mypath, column + ".png")
        fig.savefig(file_path, dpi=600)
        plt.close(fig)
        pylab.close(fig)
        del fig
    colors = sns.color_palette("Set1", n_colors=5)
    colors = colors[-4:] + colors[:-4]
    sns.set_palette(colors)

    for column in headers:
        sns.pairplot(data_frame, vars=["popindex", str(column)], kind="reg")
        fig = plt.gcf()
        fig.set_size_inches(14, 9)
        file_path = os.path.join(mypath, column + "_pair.png")
        fig.savefig(file_path, dpi=600)
        plt.close(fig)
        pylab.close(fig)
        del fig

    # plot pop index
    column = dataset.columns[-1]
    fig = plt.figure()
    x = data_frame[column].T
    sns.distplot(x)
    file_path = os.path.join(mypath, column + "_dist.png")
    fig.savefig(file_path, dpi=600)
    plt.close(fig)
    del fig
