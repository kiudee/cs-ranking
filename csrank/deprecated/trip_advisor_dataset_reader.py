import copy
import inspect
import os

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

try:
    import pandas as pd
except ImportError:
    from csrank.util import MissingExtraError

    raise MissingExtraError("pandas", "data")

# Get hotel_dataset for each city in a dictionary
POPINDEX = "popindex"
MUENCHEN = "muenchen"
HAMBURG = "hamburg"
FRANKFURT = "frankfurt"
DUSSLEDORF = "dussledorf"
BERLIN = "berlin"
DATASET_FOLDER = "hotel_dataset"

__all__ = ["TripAdviosrDataset"]


class TripAdviosrDataset(object):
    def __init__(self, **kwargs):
        self.city_names = [BERLIN, DUSSLEDORF, MUENCHEN, HAMBURG, FRANKFURT]
        self.MIN_HOTEL_RANK_LENGTH = 110
        dirname = os.path.dirname(
            os.path.abspath(inspect.getfile(inspect.currentframe()))
        )
        mypath = os.path.join(dirname, DATASET_FOLDER, "trip_advisor.csv")
        full_dataset = pd.read_csv(mypath)
        index_1 = full_dataset.columns.get_loc(BERLIN)
        city_indices = np.arange(index_1, index_1 + 5)
        self.datasets_dictionaries = dict()
        self.datasets_dictionaries[BERLIN] = (
            full_dataset[full_dataset.berlin == 1]
            .drop(full_dataset.columns[city_indices], axis=1)
            .sort_values(by=[POPINDEX], ascending=[True])
        )
        self.datasets_dictionaries[DUSSLEDORF] = (
            full_dataset[full_dataset.duesseldorf == 1]
            .drop(full_dataset.columns[city_indices], axis=1)
            .sort_values(by=[POPINDEX], ascending=[True])
        )
        self.datasets_dictionaries[FRANKFURT] = (
            full_dataset[full_dataset.frankfurt == 1]
            .drop(full_dataset.columns[city_indices], axis=1)
            .sort_values(by=[POPINDEX], ascending=[True])
        )
        self.datasets_dictionaries[HAMBURG] = (
            full_dataset[full_dataset.hamburg == 1]
            .drop(full_dataset.columns[city_indices], axis=1)
            .sort_values(by=[POPINDEX], ascending=[True])
        )
        self.datasets_dictionaries[MUENCHEN] = (
            full_dataset[full_dataset.muenchen == 1]
            .drop(full_dataset.columns[city_indices], axis=1)
            .sort_values(by=[POPINDEX], ascending=[True])
        )
        for key, value in self.datasets_dictionaries.items():
            popindex = value[POPINDEX]
            value.drop(POPINDEX, axis=1, inplace=True)
            value.insert(len(list(value.columns)), POPINDEX, popindex)
            value.reset_index(drop=True)
        self.all_headers = self.datasets_dictionaries[BERLIN].columns[0:-1]
        self.all_dataframes = pd.concat(
            list(self.datasets_dictionaries.values()), ignore_index=True
        )

    # Get features and rankings for given hotel_dataset for a city
    def data_for_city(self, city, features_to_remove=[]):

        if city not in self.city_names:
            raise ValueError(
                f"Data is not present for the city, hotel_dataset is present only for: {set(self.city_names)}"
            )
        dataframe = copy.deepcopy(self.datasets_dictionaries[city])
        dataframe.drop(features_to_remove, axis=1, inplace=True)
        headers = list(dataframe.columns[0:-1])
        # index_to_remove = [np.where(headers == f)[0] for f in features_to_remove]
        array = dataframe.as_matrix()
        features = array[:, 0:-1]
        # get actual and normalized ranks
        actual_ranks = array[:, -1]
        scaler = MinMaxScaler()
        normalized_ranks = np.array(actual_ranks)
        normalized_ranks = scaler.fit_transform(normalized_ranks.reshape(-1, 1)).T[0]
        # get features and filtered features
        features = np.array(
            [np.log(np.array(features[:, i]) + 1) for i in range(len(features[0]))]
        )
        features = np.array(features.T)
        scaler = StandardScaler()
        features = scaler.fit_transform(features)
        actual_ranks = np.arange(len(actual_ranks))

        features = features[None, :]
        actual_ranks = actual_ranks[None, :]
        normalized_ranks = normalized_ranks[None, :]

        return headers, features, actual_ranks, normalized_ranks

    def data_for_cities(self, cities=[BERLIN], features_to_remove=[]):
        X = []
        Y = []
        norm_ranks = []
        for city in cities:
            headers, features, actual_ranks, normalized_ranks = self.data_for_city(
                city, features_to_remove
            )
            X.append(np.array(features[0]))
            Y.append(np.array(actual_ranks[0]))
            norm_ranks.append((np.array(normalized_ranks[0])))
        X = np.array(X)
        Y = np.array(Y)
        norm_ranks = np.array(norm_ranks)
        return X, Y, norm_ranks, headers
