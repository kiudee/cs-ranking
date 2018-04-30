import logging
from itertools import combinations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state

from csrank.labelranking.label_ranker import LabelRanker
from csrank.tunable import Tunable
from csrank.util import print_dictionary


class RankingbyPairwiseComparisonLabelRanker(LabelRanker, Tunable):
    _tunable = None

    def __init__(self, n_features, C=1, tol=1e-4, normalize=True, fit_intercept=True, random_state=None, **kwargs):
        self.normalize = normalize
        self.n_features = n_features
        self.C = C
        self.tol = tol
        self.logger = logging.getLogger('RPC')
        self.random_state = check_random_state(random_state)
        self.fit_intercept = fit_intercept

    def get_model_for_pair(self, X, Y, pair):
        Y_train = []
        for ranking in Y:
            rank1 = ranking[pair[0]]
            rank2 = ranking[pair[1]]
            if rank1 > rank2:
                Y_train.append(1)
            else:
                Y_train.append(0)
        model = LogisticRegression(C=self.C, tol=self.tol, fit_intercept=self.fit_intercept,
            random_state=self.random_state)
        model.fit(X, Y_train)
        return model

    def fit(self, X, Y, **kwargs):
        if (self.normalize):
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
        N, self.n_labels = Y.shape
        labels = np.arange(self.n_labels)
        self.models_with_pairwise_preferences = list()

        for pair in list(combinations(labels, 2)):
            pair_rank_and_model = []
            model = self.get_model_for_pair(X, Y, pair)
            pair_rank_and_model.append(pair)
            pair_rank_and_model.append(model)
            self.models_with_pairwise_preferences.append(pair_rank_and_model)

        self.logger.debug('Finished Creating the model, now fitting started')

    def predict_scores(self, X, **kwargs):
        # nearest neighbour model get the neighbouring rankings
        scores = []
        for context in X:
            label_scores = np.zeros(self.n_labels)
            for pair_rank_and_model in self.models_with_pairwise_preferences:
                score = pair_rank_and_model[1].predict_proba(context[None, :]).flatten()[0]
                label_scores[pair_rank_and_model[0][0]] = label_scores[pair_rank_and_model[0][0]] + score
                label_scores[pair_rank_and_model[0][1]] = label_scores[pair_rank_and_model[0][1]] + (1 - score)
            scores.append(label_scores)
        return np.array(scores)

    def predict(self, X, **kwargs):
        return LabelRanker.predict(self, X, **kwargs)

    def set_tunable_parameters(self, C=1, tol=1e-4, **point):
        self.tol = tol
        self.C = C
        if len(point) > 0:
            self.logger.warning('This ranking algorithm does not support'
                                ' tunable parameters'
                                ' called: {}'.format(print_dictionary(point)))
