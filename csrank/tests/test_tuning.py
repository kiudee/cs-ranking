import os

import numpy as np
import pytest
from keras.metrics import binary_accuracy

from csrank.tuning import check_learner_class
from ..tuning import ParameterOptimizer

OPTIMIZER_PATH = os.path.join(os.getcwd(), 'opt')

n_instances = 500
n_features = 3


@pytest.fixture
def optimizer():
    from ..tunable import Tunable

    class RankerStub(Tunable):

        def fit(self, X, Y, **kwargs):
            self.seed = int(np.sum(list(self.__dict__.values())))

        def predict(self, X, **kwargs):
            random_state = np.random.RandomState(self.seed)
            weight = random_state.rand(n_features, 2)
            scores = np.dot(X, weight) / np.dot(X, weight).sum(axis=1)[:, None]
            return scores.argmax(axis=1)

        def set_tunable_parameters(self, **point):
            self.__dict__.update(point)

        def __call__(self, X, *args, **kwargs):
            return self.predict(X, **kwargs)

    ranker = RankerStub()

    rankers = [RankerStub() for _ in range(2)]
    test_params = {
        rankers[0]: dict(a=(1.0, 4.0)),
        ranker: dict(b=(4.0, 7.0), c=(7.0, 10.0)),
        rankers[1]: dict(d=(10.0, 13.0))
    }

    opt = ParameterOptimizer(
        learner=ranker,
        optimizer_path=OPTIMIZER_PATH,
        tunable_parameter_ranges=test_params,
        ranker_params=dict(),
        validation_loss=binary_accuracy
    )
    return opt, rankers, test_params


@pytest.fixture(scope="module")
def trivial_ranking_problem():
    random_state = np.random.RandomState(123)
    x = random_state.rand(n_instances, n_features)
    y_true = random_state.randint(0, 2, n_instances)
    return x, y_true


def check(value, bound):
    if bound[0] <= value <= bound[1]:
        return True
    return False


def test_parameter_optimizer(trivial_ranking_problem, optimizer):
    x, y = trivial_ranking_problem
    opt, rankers, test_params = optimizer
    opt.fit(x, y, n_iter=10)
    os.remove(OPTIMIZER_PATH)
    for point in opt.opt.Xi:
        i = 0
        for param_ranges in test_params.values():
            for key, bound in param_ranges.items():
                assert check(point[i], bound)
                i += 1


def test_set_parameters(optimizer):
    opt, rankers, test_params = optimizer
    point = list(range(4))
    opt._set_new_parameters(point)
    i = 0
    for ranker in test_params.keys():
        try:
            check_learner_class(ranker)
        except AttributeError as exc:
            pytest.fail(exc, pytrace=True)
        for param in test_params[ranker]:
            assert ranker.__dict__[param] == point[i]
            i += 1
