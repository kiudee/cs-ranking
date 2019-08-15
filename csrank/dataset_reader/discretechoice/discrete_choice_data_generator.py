import numpy as np
from pygmo import hypervolume
from scipy.spatial.distance import squareform, pdist
from sklearn.datasets import make_regression, make_blobs
from sklearn.gaussian_process.kernels import Matern
from sklearn.utils import check_random_state

from csrank.constants import DISCRETE_CHOICE
from .util import convert_to_label_encoding
from ..synthetic_dataset_generator import SyntheticDatasetGenerator
from ..util import create_pairwise_prob_matrix


class DiscreteChoiceDatasetGenerator(SyntheticDatasetGenerator):
    def __init__(self, dataset_type='medoid', **kwargs):
        super(DiscreteChoiceDatasetGenerator, self).__init__(learning_problem=DISCRETE_CHOICE, **kwargs)
        dataset_function_options = {'linear': self.make_linear_transitive,
                                    'medoid': self.make_intransitive_medoids,
                                    'nearest_neighbour_medoid': self.nearest_neighbour,
                                    'hypervolume': self.make_hv_dataset,
                                    'gp_transitive': self.make_gp_transitive,
                                    'gp_non_transitive': self.make_gp_non_transitive}
        if dataset_type not in dataset_function_options.keys():
            dataset_type = "medoid"
        self.logger.info("dataset type {}".format(dataset_type))
        self.dataset_function = dataset_function_options[dataset_type]

    def make_linear_transitive(self, n_instances=1000, n_objects=5, noise=0.0, n_features=100, n_informative=10,
                               seed=42, **kwd):
        random_state = check_random_state(seed=seed)
        X, y, coeff = make_regression(n_samples=n_instances * n_objects, n_features=n_features,
                                      n_informative=n_informative, coef=True, noise=noise, random_state=random_state)
        X = X.reshape(n_instances, n_objects, n_features)
        y = y.reshape(n_instances, n_objects)
        Y = y.argmax(axis=1)
        Y = convert_to_label_encoding(Y, n_objects)
        return X, Y

    def make_intransitive_medoids(self, n_instances=100, n_objects=5, n_features=100, seed=42, **kwd):
        random_state = check_random_state(seed=seed)
        X = random_state.uniform(size=(n_instances, n_objects, n_features))
        Y = np.empty(n_instances)
        for i in range(n_instances):
            D = squareform(pdist(X[i], metric='euclidean'))
            sum_dist = D.mean(axis=0)
            medoid = np.argmin(sum_dist)
            Y[i] = medoid
        X = np.array(X)
        Y = np.array(Y)
        Y = convert_to_label_encoding(Y, n_objects)
        return X, Y

    def nearest_neighbour(self, n_instances=100, n_objects=5, n_features=100, seed=42, **kwd):
        random_state = check_random_state(seed=seed)
        X = random_state.uniform(size=(n_instances, n_objects, n_features))
        Y = np.empty(n_instances)
        for i in range(n_instances):
            D = squareform(pdist(X[i], metric='euclidean'))
            sum_dist = D.mean(axis=0)
            medoid = np.argmin(sum_dist)
            Y[i] = np.argsort(D[medoid])[1]
        X = np.array(X)
        Y = np.array(Y)
        Y = convert_to_label_encoding(Y, n_objects)
        return X, Y

    def make_hv_dataset(self, n_instances=1000, n_objects=5, n_features=5, seed=42, cluster_spread=1.0, **kwd):
        def sample_unit_ball(n_f=2, rng=None, radius=1.):
            rng = check_random_state(rng)
            X = rng.randn(1, n_f)
            u = rng.uniform(size=1)[:, None]
            X /= np.linalg.norm(X, axis=1, ord=2)[:, None]
            X *= radius * u
            return X[0]

        random_state = check_random_state(seed=seed)
        X = random_state.rand(n_instances, n_objects, n_features)
        # Normalize to unit circle and fold to lower quadrant
        X = -np.abs(X / np.sqrt(np.power(X, 2).sum(axis=2))[..., None])
        Y = np.empty(n_instances, dtype=int)
        for i in range(n_instances):
            center = sample_unit_ball(n_f=n_features, rng=i, radius=cluster_spread)
            X[i] = X[i] + center
            hv = hypervolume(X[i])
            cont = hv.contributions(center)
            Y[i] = np.argmax(cont)
        Y = convert_to_label_encoding(Y, n_objects)
        return X, Y

    def make_gp_transitive(self, n_instances=1000, n_objects=5, noise=0.0, n_features=100, kernel_params=None, seed=42,
                           **kwd):
        """Creates a nonlinear object ranking problem by sampling from a
        Gaussian process as the latent utility function.
        Note that this function needs to compute a kernel matrix of size
        (n_instances * n_objects) ** 2, which could allocate a large chunk of the
        memory."""
        random_state = check_random_state(seed=seed)

        if kernel_params is None:
            kernel_params = dict()
        n_total = n_instances * n_objects
        X = random_state.rand(n_total, n_features)
        L = np.linalg.cholesky(Matern(**kernel_params)(X))
        f = (L.dot(random_state.randn(n_total)) +
             random_state.normal(scale=noise, size=n_total))
        X = X.reshape(n_instances, n_objects, n_features)
        f = f.reshape(n_instances, n_objects)
        Y = f.argmax(axis=1)
        Y = convert_to_label_encoding(Y, n_objects)
        return X, Y

    def make_gp_non_transitive(self, n_instances=1000, n_objects=5, n_features=100, center_box=(-10.0, 10.0),
                               cluster_std=2.0, seed=42, **kwd):
        n_samples = n_instances * n_objects
        random_state = check_random_state(seed=seed)
        x, y = make_blobs(n_samples=n_samples, centers=n_objects, n_features=n_features, cluster_std=cluster_std,
                          center_box=center_box, random_state=random_state, shuffle=True)
        y = np.array([y])
        samples = np.append(x, y.T, axis=1)
        samples = samples[samples[:, n_features].argsort()]
        pairwise_prob = create_pairwise_prob_matrix(n_objects)
        X = []
        Y = []
        for inst in range(n_instances):
            feature = np.array([samples[inst + i * n_instances, 0:-1] for i in range(n_objects)])
            matrix = np.random.binomial(1, pairwise_prob)
            choice = np.argmax(np.sum(matrix, axis=1))
            X.append(feature)
            Y.append(choice)
        X = np.array(X)
        Y = np.array(Y)
        Y = convert_to_label_encoding(Y, n_objects)
        return X, Y

    def get_single_train_test_split(self):
        return super(DiscreteChoiceDatasetGenerator, self).get_single_train_test_split()

    def get_train_test_datasets(self, n_datasets=5):
        return super(DiscreteChoiceDatasetGenerator, self).get_train_test_datasets(n_datasets=n_datasets)

    def get_dataset_dictionaries(self, lengths=[5, 6]):
        return super(DiscreteChoiceDatasetGenerator, self).get_dataset_dictionaries(lengths=lengths)
