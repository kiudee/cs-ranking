import numpy as np

from csrank import ChoiceDatasetGenerator


def test_pareto_problem_generation():
    """A simple sanity check for Pareto problem generation."""
    gen = ChoiceDatasetGenerator(
        dataset_type="pareto",
        random_state=42,
        n_train_instances=11,
        n_test_instances=1,
        n_objects=3,
        n_features=2,
    )
    X_train, Y_train, X_test, Y_test = gen.get_single_train_test_split()
    assert X_train.shape == (11, 3, 2)
    assert Y_train.shape == (11, 3)
    assert X_test.shape == (1, 3, 2)

    def is_binary_array(a):
        return np.logical_or(a == 0, a == 1).all()

    assert is_binary_array(Y_train)
    assert is_binary_array(Y_test)
