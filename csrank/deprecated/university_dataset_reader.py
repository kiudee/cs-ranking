import inspect
import os

import numpy as np
import sklearn.preprocessing as sklearn_preprocessing

try:
    import pandas as pd
except ImportError:
    from csrank.util import MissingExtraError

    raise MissingExtraError("pandas", "data")

DATASET_FOLDER = "university_dataset"
UNIVERSITY_ = "university_"
WORLD_RANK = "world_rank"
COUNTRY = "country"
UNIVERSITY_NAME = "university_name"
years = [2012, 2013, 2014, 2015]
MIN_UNIV_RANK_LENGTH = 95

__all__ = ["univerisity_dataset_for_year", "MIN_UNIV_RANK_LENGTH", "years"]


def num(s):
    try:
        return float(s)
    except ValueError:
        try:
            return int(s)
        except ValueError:
            return s


def cal_mean(w):
    try:
        if "-" in w:
            return (float(str(w).split("-")[0]) + float(str(w).split("-")[1])) / 2
        else:
            return num(str(w))
    except TypeError:
        return num(str(w))


def create_joined_dataset():
    dir_name = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    datasetCwur = pd.read_csv(
        os.path.join(dir_name, DATASET_FOLDER, "cwurData.csv")
    ).rename(columns={"institution": UNIVERSITY_NAME})
    datasetTimes = pd.read_csv(os.path.join(dir_name, DATASET_FOLDER, "timesData.csv"))
    datasetShanghai = pd.read_csv(
        os.path.join(dir_name, DATASET_FOLDER, "shanghaiData.csv")
    )
    columns = np.concatenate(
        (
            datasetCwur.columns[3:-1],
            datasetTimes.columns[3:-1],
            datasetShanghai.columns[3:-1],
        )
    ).flatten()
    columns = np.insert(columns, 0, UNIVERSITY_NAME)
    columns = np.insert(columns, 0, COUNTRY)
    columns = np.insert(columns, len(columns), WORLD_RANK)
    # columns = np.insert(columns, len(columns), WORLD_RANK+'_array')
    for year in years:
        df1 = datasetCwur.loc[datasetCwur["year"] == year]
        df2 = datasetTimes.loc[datasetTimes["year"] == year]
        df3 = datasetShanghai.loc[datasetShanghai["year"] == year]
        list1 = list(df1[UNIVERSITY_NAME])
        list2 = list(df2[UNIVERSITY_NAME])
        list3 = list(df3[UNIVERSITY_NAME])
        names = list(set().union(list1, list2, list3))
        dataYear = []

        for name in names:
            row1 = df1.loc[df1[UNIVERSITY_NAME] == name]
            row2 = df2.loc[df2[UNIVERSITY_NAME] == name]
            row3 = df3.loc[df3[UNIVERSITY_NAME] == name]
            a = len(row1) != 0
            b = len(row2) != 0
            c = len(row3) != 0
            conditions = np.array([a, b, c])

            if np.sum(conditions) >= 2:
                country = "none"
                world_ranks = []
                if b:
                    country = row2.iloc[0][2]
                if a:
                    country = row1.iloc[0][2]
                oneRow = np.array([name, country])
                if a:
                    row = [num(s) for s in row1.iloc[0][3:-1]]
                    oneRow = np.append(oneRow, row)
                    world_ranks.append(row1.iloc[0][0])
                else:
                    zeros = np.zeros(10)
                    zeros[:] = np.inf
                    oneRow = np.append(oneRow, zeros)

                if b:
                    row = [num(s) for s in row2.iloc[0][3:-1]]
                    row[-2] = float(str(row[-2]).split("%")[0])
                    try:
                        row[-1] = float(str(row[-1]).split(" : ")[0]) / float(
                            str(row[-1]).split(" : ")[1]
                        )
                    except IndexError:
                        row[-1] = np.inf
                    except ZeroDivisionError:
                        row[-1] = 1.0
                    oneRow = np.append(oneRow, row)
                    world_ranks.append(row2.iloc[0][0])
                else:
                    zeros = np.zeros(10)
                    zeros[:] = np.inf
                    oneRow = np.append(oneRow, zeros)

                if c:
                    row = [num(s) for s in row3.iloc[0][3:-1]]
                    oneRow = np.append(oneRow, row)
                    world_ranks.append(row3.iloc[0][0])

                else:
                    zeros = np.zeros(7)
                    zeros[:] = np.inf
                    oneRow = np.append(oneRow, zeros)
                world_rank = np.mean(np.array([cal_mean(w) for w in world_ranks]))
                oneRow = np.append(oneRow, [world_rank])
                # oneRow = np.append(oneRow, str(world_ranks))
                dataYear.append(oneRow)
        dataframeYear = pd.DataFrame(dataYear, columns=columns)
        dataframeYear = dataframeYear.convert_objects(convert_numeric=True)
        total_score = np.array(dataframeYear[["total_score", "score"]])[:, 0:2]
        total_score[np.isnan(total_score)] = 0.0
        total_score[np.isinf(total_score)] = 0.0
        dataframeYear["total_score"] = np.mean(total_score, axis=1)
        dataframeYear.drop("score", axis=1, inplace=True)
        dataframeYear.drop("broad_impact", axis=1, inplace=True)
        dataframeYear = dataframeYear.sort_values(by=[WORLD_RANK], ascending=[True])
        dataframeYear.to_csv(
            os.path.join(dir_name, DATASET_FOLDER, UNIVERSITY_ + str(year) + ".csv")
        )


# Get features and rankings for given hotel_dataset for a city
def univerisity_dataset_for_year(year):
    dir_name = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

    if int(year) not in years:
        raise ValueError(
            f"Data is not present for the year, university dataset is present only for: {set(years)}"
        )
    path = os.path.join(dir_name, DATASET_FOLDER, UNIVERSITY_ + str(year) + ".csv")
    dataframe = pd.read_csv(path)
    headers = np.array(dataframe.columns)
    headers[1], headers[2] = headers[2], headers[1]
    dataframe = dataframe[headers]
    size = len(headers)
    for cat_cols in dataframe.select_dtypes(["object"]).columns:
        dataframe[cat_cols] = dataframe[cat_cols].astype("category")
        dataframe[cat_cols] = pd.Categorical.from_array(dataframe[cat_cols]).codes
        dataframe[cat_cols] = dataframe[cat_cols].replace(-1, np.NaN)

    array = dataframe.as_matrix()
    features = array[:, 2 : size - 1]
    headers = headers[2 : size - 1]
    for i in range(len(headers)):
        col = features[:, i]
        non_nans = [f for f in col if not (np.isinf(f) or np.isnan(f))]
        mean = np.nanmean(non_nans)
        col[np.isnan(col)] = mean
        col[np.isinf(col)] = mean

    # if (np.isnan(np.sum(features))):
    #     imp = sklearn_preprocessing.Imputer()
    #     features = imp.fit_transform(features).T
    actual_ranks = array[:, size - 1]
    scaler = sklearn_preprocessing.MinMaxScaler()
    normalized_ranks = np.array(actual_ranks)
    normalized_ranks = scaler.fit_transform(normalized_ranks.reshape(-1, 1)).T[0]
    # get features and filtered features
    features = np.array(
        [np.log(np.array(features[:, i]) + 1) for i in range(len(features[0]))]
    )
    scaler = sklearn_preprocessing.StandardScaler()
    features = scaler.fit_transform(features).T
    actual_ranks = np.arange(len(actual_ranks))

    features = features[None, :]
    actual_ranks = actual_ranks[None, :]
    normalized_ranks = normalized_ranks[None, :]

    return headers, features, actual_ranks, normalized_ranks
