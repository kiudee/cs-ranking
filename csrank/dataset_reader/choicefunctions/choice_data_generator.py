import numpy as np
from sklearn.utils import check_random_state

from csrank.constants import CHOICE_FUNCTION
from ..synthetic_dataset_generator import SyntheticDatasetGenerator


class ChoiceDatasetGenerator(SyntheticDatasetGenerator):

    def __init__(self, dataset_type='pareto', **kwargs):
        super(ChoiceDatasetGenerator, self).__init__(
            learning_problem=CHOICE_FUNCTION, **kwargs)
        dataset_function_options = {'linear': self.make_latent_linear_choices,
                                    "pareto": self.make_globular_pareto_choices}
        if dataset_type not in dataset_function_options.keys():
            dataset_type = "pareto"
        self.dataset_function = dataset_function_options[dataset_type]

    def get_single_train_test_split(self):
        return super(ChoiceDatasetGenerator, self).get_single_train_test_split()

    def get_train_test_datasets(self, n_datasets=5):
        return super(ChoiceDatasetGenerator, self).get_train_test_datasets(n_datasets=n_datasets)

    def make_globular_pareto_choices(self, n_instances=10000, n_features=2, n_objects=10, seed=42,
                                     cluster_spread=1., cluster_size=10, **kwargs):
        def pareto_front(X, signs=None):
            n_points, n_attributes = X.shape
            if signs is None:
                signs = -np.ones(n_attributes)
            pareto = np.ones(n_points, dtype=bool)
            for i, attr in enumerate(X):
                pareto[i] = np.all(np.any((X * signs[None, :]) <= (attr * signs), axis=1))
            return pareto

        def sample_unit_ball(n_inst=10000, n_features=2, rng=None, radius=1.):
            rng = check_random_state(rng)
            X = rng.randn(n_inst, n_features)
            u = rng.uniform(size=n_inst)[:, None]
            X /= np.linalg.norm(X, axis=1, ord=2)[:, None]
            X *= radius * u
            return X

        def make_randn_pareto_choices(n_instances=10000, n_features=2, n_objects=10, data_seed=None, center=0.):
            """Generate random objects from a d-dimensional isometric normal distribution.

            This should be the easiest possible Pareto-problem, since the model can learn
            a latent-utility which scores how likely a point is on the front (independent
            of the other points)."""
            rand = check_random_state(data_seed)
            X = rand.randn(n_instances, n_objects, n_features)
            Y = np.empty((n_instances, n_objects), dtype=bool)
            for i in range(n_instances):
                Y[i] = pareto_front(X[i])
            return X + center, Y

        rand = check_random_state(seed)
        X = np.empty((n_instances, n_objects, n_features))
        Y = np.empty((n_instances, n_objects), dtype=int)
        for i in range(int(n_instances / cluster_size)):
            center = sample_unit_ball(n_inst=1, n_features=n_features,
                                      rng=rand, radius=cluster_spread)
            x, y = make_randn_pareto_choices(n_instances=cluster_size, n_features=n_features, n_objects=n_objects,
                                             data_seed=rand, center=center)
            X[i * cluster_size:(i + 1) * cluster_size] = x
            Y[i * cluster_size:(i + 1) * cluster_size] = y
        return X, Y

    def make_latent_linear_choices(self, n_instances=10000, n_features=2, n_objects=6, n_rep_units=5, threshold=0.,
                                   seed=42, **kwargs):
        rand = check_random_state(seed)
        ranw = check_random_state(rand.randint(2 ** 32, dtype='uint32'))
        X = rand.uniform(-1, 1, size=(n_instances, n_objects, n_features))
        W_rep = ranw.randn(n_features, n_rep_units)
        rep = X.dot(W_rep).mean(axis=-2)
        w_join = ranw.randn(n_features + n_rep_units)
        joint_matrix = np.empty((n_objects, n_instances, n_features + n_rep_units), dtype=np.float32)
        for i in range(n_objects):
            joint_matrix[i] = np.concatenate((X[:, i], rep), axis=-1)
        scores = joint_matrix.dot(w_join)
        Y = scores > threshold
        Y = Y.astype(int)
        return X, Y.T
