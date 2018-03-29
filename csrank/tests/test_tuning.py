from pytest import fixture
from ..tuning import ParameterOptimizer


@fixture
def optimizer():
    from ..tunable import Tunable

    class RankerStub(Tunable):
        def set_tunable_parameters(self, point):
            self.__dict__.update(point)

    rankers = [RankerStub() for _ in range(3)]
    test_params = {
        rankers[0]: dict(a=(0, 3)),
        rankers[1]: dict(b=(0, 3), c=(0, 3)),
        rankers[2]: dict(d=(0, 3))
    }

    opt = ParameterOptimizer(
        ranker=RankerStub(),
        optimizer_path='./',
        tunable_parameter_ranges=test_params,
        ranker_params=dict()
    )
    return opt, rankers, test_params


def test_parameter_optimizer():
    assert True


def test_set_parameters(optimizer):
    opt, rankers, test_params = optimizer
    point = list(range(4))
    opt._set_new_parameters(point)
    i = 0
    for ranker in test_params.keys():
        for param in test_params[ranker]:
            assert ranker.__dict__[param] == point[i]
            i += 1
