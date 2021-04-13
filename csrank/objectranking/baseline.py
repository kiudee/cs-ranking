import logging

from sklearn.utils import check_random_state

from csrank.learner import Learner
from .object_ranker import ObjectRanker

logger = logging.getLogger(__name__)


class RandomBaselineRanker(ObjectRanker, Learner):
    def __init__(self, random_state=None, **kwargs):
        """
        Baseline assigns the average number of chosen objects in the given choice sets and chooses all the objects.

        :param kwargs: Keyword arguments for the algorithms
        """

        self.random_state = random_state

    def _pre_fit(self):
        super()._pre_fit()
        self.random_state_ = check_random_state(self.random_state)

    def fit(self, X, Y, **kwd):
        self._pre_fit()
        return self

    def _predict_scores_fixed(self, X, **kwargs):
        n_instances, n_objects, n_features = X.shape
        return self.random_state_.rand(n_instances, n_objects)
