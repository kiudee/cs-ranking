import logging
from collections import OrderedDict
from itertools import combinations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state

from csrank.labelranking.label_ranker import LabelRanker
from csrank.tunable import Tunable
from csrank.util import tunable_parameters_ranges


class RankingbyPairwiseComparison(LabelRanker, Tunable):
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

    @classmethod
    def set_tunable_parameter_ranges(cls, param_ranges_dict):
        logger = logging.getLogger('RPC')
        return tunable_parameters_ranges(cls, logger, param_ranges_dict)

    def set_tunable_parameters(self, params):
        self.logger.debug('Got the following parameter vector: {}'.format(params))
        named = dict(zip(self._tunable.keys(), params))
        for name, param in named.items():
            if name == 'C':
                self.C = param
            elif name == 'tolerance':
                self.tol = param
            else:
                self.logger.warning('This ranking algorithm does not support'
                                    'a tunable parameter called {}'.format(name))

    @classmethod
    def tunable_parameters(cls):
        if cls._tunable is None:
            cls._tunable = OrderedDict([
                ('C', (1, 12)),
                ('tolerance', (1e-4, 5e-1, "log-uniform"))
            ])
        return list(cls._tunable.values())

# if __name__ == '__main__':
#     log_path = os.path.join(os.getcwd(), 'logs', "rpc_exp.log")
#     create_dir_recursively(log_path, True)
#     log_path = rename_file_if_exist(log_path)
#     logging.basicConfig(filename=log_path, level=logging.DEBUG,
#                         format='%(asctime)s %(name)s %(levelname)-8s %(message)s',
#                         datefmt='%Y-%m-%d %H:%M:%S')
#     logger = logging.getLogger(name='Experiment')
#
#     I = IntelligentSystemGroupDatasetReader()
#     X, Y = I.get_complete_dataset()
#     skf = KFold(n_splits=10, shuffle=True, random_state=42)
#     tau = []
#     spr = []
#     for train_indicies, test_indicies in list(skf.split(Y)):
#         train = X[train_indicies]
#         train_orderings = Y[train_indicies]
#         test = X[test_indicies]
#         test_orderings = Y[test_indicies]
#         n_features = X.shape[1]
#         iblr = RankingbyPairwiseComparison(n_features=n_features)
#         iblr.fit(train, train_orderings)
#
#         predicted = iblr(test)
#         tau.append(kendalls_mean(predicted, test_orderings))
#         spr.append(spearman_mean(predicted, test_orderings))
#     tau = np.array(tau)
#     spr = np.array(spr)
#     logging.info("****************Kendalls Tau******************")
#     logging.info("Kendalls: " + repr(tau))
#     logging.info("Mean: " + repr(np.mean(tau)))
#     logging.info("Std: " + repr(np.std(tau)))
#     logging.info("****************Spearman Corr******************")
#     logging.info("Spearman: " + repr(spr))
#     logging.info("Mean: " + repr(np.mean(spr)))
#     logging.info("Std: " + repr(np.std(spr)))
