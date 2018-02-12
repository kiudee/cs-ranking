import logging
from collections import OrderedDict

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state

from csrank.labelranking.label_ranker import LabelRanker
from csrank.labelranking.placketluce_model import get_pl_parameters_for_rankings
from csrank.tunable import Tunable
from csrank.util import ranking_ordering_conversion, tunable_parameters_ranges


class InstanceBaseLabelRanking(LabelRanker, Tunable):
    _tunable = None

    def __init__(self, n_features, k=20, algorithm="ball_tree", normalize=True, random_state=None, **kwargs):
        self.normalize = normalize
        self.n_features = n_features
        self.k = k
        self.algorithm = algorithm
        self.logger = logging.getLogger('InstanceBasedLabelRanker')
        self.random_state = check_random_state(random_state)

    def fit(self, X, Y, **kwargs):
        self.model = NearestNeighbors(n_neighbors=self.k, algorithm=self.algorithm)

        if (self.normalize):
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
        self.logger.debug('Finished Creating the model, now fitting started')
        self.model.fit(X)
        self.train_orderings = Y

    def predict_scores(self, X, **kwargs):
        # nearest neighbour model get the neighbouring rankings
        distances, indices = self.model.kneighbors(X)
        scores = []
        for i in range(len(X)):
            # Get all the neighbouring ranks and fir the pl model on it
            rankings = ranking_ordering_conversion(self.train_orderings[indices[i]])
            parameters = get_pl_parameters_for_rankings(rankings)
            scores.append(parameters)
        return np.array(scores)

    def predict(self, X, **kwargs):
        return LabelRanker.predict(self, X, **kwargs)

    @classmethod
    def set_tunable_parameter_ranges(cls, param_ranges_dict):
        logger = logging.getLogger('InstanceBasedLabelRanker')
        return tunable_parameters_ranges(cls, logger, param_ranges_dict)

    def set_tunable_parameters(self, params):
        self.logger.debug('Got the following parameter vector: {}'.format(params))
        named = dict(zip(self._tunable.keys(), params))
        for name, param in named.items():
            if name == 'k':
                self.k = param
            elif name == 'algorithm':
                self.algorithm = param
            else:
                self.logger.warning('This ranking algorithm does not support'
                                    'a tunable parameter called {}'.format(name))

    @classmethod
    def tunable_parameters(cls):
        if cls._tunable is None:
            cls._tunable = OrderedDict([
                ('k', (1, 12)),
                ('algorithm', ("ball_tree", "kd_tree"))
            ])
        return list(cls._tunable.values())
