import logging
from itertools import combinations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state

from csrank.labelranking.label_ranker import LabelRanker
from csrank.learner import Learner
from csrank.numpy_util import scores_to_rankings
from csrank.util import print_dictionary


class RankingbyPairwiseComparisonLabelRanker(LabelRanker, Learner):

    def __init__(self, n_features, C=1, tol=1e-4, normalize=True, fit_intercept=True, random_state=None, **kwargs):
        self.normalize = normalize
        self.n_features = n_features
        self.C = C
        self.tol = tol
        self.logger = logging.getLogger('RPC')
        self.random_state = check_random_state(random_state)
        self.fit_intercept = fit_intercept
        self.models_with_pairwise_preferences = list()
        self.n_labels = None

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
        if self.normalize:
            std_scalar = StandardScaler()
            X = std_scalar.fit_transform(X)
        N, self.n_labels = Y.shape
        labels = np.arange(self.n_labels)
        for pair in list(combinations(labels, 2)):
            pair_rank_and_model = []
            model = self.get_model_for_pair(X, Y, pair)
            pair_rank_and_model.append(pair)
            pair_rank_and_model.append(model)
            self.models_with_pairwise_preferences.append(pair_rank_and_model)

        self.logger.debug('Finished Creating the model, now fitting started')

    def _predict_scores_fixed(self, X, **kwargs):
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

    def predict_scores(self, X, **kwargs):
        return self._predict_scores_fixed(X, **kwargs)

    def predict_for_scores(self, scores, **kwargs):
        self.logger('Predicting rankings')
        return scores_to_rankings(scores)

    def predict(self, X, **kwargs):
        return super().predict(self, X, **kwargs)

    def clear_memory(self, **kwargs):
        self.logger.info("Clearing memory")
        pass

    def set_tunable_parameters(self, C=1, tol=1e-4, **point):
        self.tol = tol
        self.C = C
        if len(point) > 0:
            self.logger.warning('This ranking algorithm does not support'
                                ' tunable parameters'
                                ' called: {}'.format(print_dictionary(point)))
