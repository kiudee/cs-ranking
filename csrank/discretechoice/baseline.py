import logging

from csrank.learner import Learner
from sklearn.utils import check_random_state

from .discrete_choice import DiscreteObjectChooser


class RandomBaselineDC(DiscreteObjectChooser, Learner):
    def __init__(self, random_state=None, **kwargs):
        """
            Baseline assigns the average number of chosen objects in the given choice sets and chooses all the objects.

            :param kwargs: Keyword arguments for the algorithms
        """

        self.logger = logging.getLogger(RandomBaselineDC.__name__)
        self.random_state = check_random_state(random_state)
        self.model = None

    def fit(self, X, Y, **kwd):
        pass

    def _predict_scores_fixed(self, X, **kwargs):
        n_instances, n_objects, n_features = X.shape
        return self.random_state.rand(n_instances, n_objects)

    def predict_scores(self, X, **kwargs):
        return super().predict_scores(X, **kwargs)

    def predict_for_scores(self, scores, **kwargs):
        return DiscreteObjectChooser.predict_for_scores(self, scores, **kwargs)

    def predict(self, X, **kwargs):
        return super().predict(X, **kwargs)

    def predict(self, X, **kwargs):
        return super().predict(X, **kwargs)

    def set_tunable_parameters(self, **point):
        pass
