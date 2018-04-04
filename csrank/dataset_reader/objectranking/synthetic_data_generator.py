import numpy as np
from pygmo import hypervolume
from scipy.spatial.distance import pdist, squareform
from sklearn.datasets import make_regression
from sklearn.datasets.samples_generator import make_blobs
from sklearn.gaussian_process.kernels import Matern
from sklearn.utils import check_random_state

from csrank.constants import OBJECT_RANKING
from csrank.dataset_reader.dataset_reader import DatasetReader
from csrank.util import scores_to_rankings, create_pairwise_prob_matrix, \
    quicksort


class SyntheticDatasetGenerator(DatasetReader):
    def __init__(self, dataset_type='medoid', n_train_instances=10000,
                 n_test_instances=10000, random_state=None,
                 **kwargs):
        super(SyntheticDatasetGenerator, self).__init__(
            learning_problem=OBJECT_RANKING, dataset_folder=None, **kwargs)
        self.random_state = check_random_state(random_state)
        dataset_function_options = {'linear': self.make_linear_transitive,
                                    'medoid': self.make_intransitive_medoids,
                                    'gp_transitive': self.make_gp_transitive,
                                    'gp_non_transitive': self.make_gp_non_transitive,
                                    "hyper_volume": self.make_hv_dataset}
        if dataset_type not in dataset_function_options.keys():
            dataset_type = "medoid"
        self.dataset_function = dataset_function_options[dataset_type]
        self.kwargs = kwargs
        self.dr_logger.info("Key word arguments {}".format(kwargs))
        self.n_train_instances = n_train_instances
        self.n_test_instances = n_test_instances

    def __load_dataset__(self):
        pass

    def splitter(self, iter):
        for i in iter:
            X_train, Y_train = self.dataset_function(**self.kwargs,
                                                     n_instances=self.n_train_instances,
                                                     seed=10 * i + 32)
            X_test, Y_test = self.dataset_function(**self.kwargs,
                                                   n_instances=self.n_test_instances,
                                                   seed=10 * i + 32)
        yield X_train, Y_train, X_test, Y_test

    def get_complete_dataset(self):
        pass

    def get_train_test_datasets(self, n_datasets=5):
        splits = np.array(n_datasets)
        return self.splitter(splits)

    def get_single_train_test_split(self):
        seed = self.random_state.randint(2 ** 32, dtype='uint32')
        self.X, self.rankings = X_train, Y_train = self.dataset_function(
            **self.kwargs,
            n_instances=self.n_train_instances, seed=seed)
        self.__check_dataset_validity__()

        seed = self.random_state.randint(2 ** 32, dtype='uint32')
        self.X, self.rankings = X_test, Y_test = self.dataset_function(
            **self.kwargs, n_instances=self.n_test_instances,
            seed=seed)
        self.__check_dataset_validity__()
        return X_train, Y_train, X_test, Y_test

    def make_linear_transitive(self, n_instances=1000, n_objects=5, noise=0.0,
                               n_features=100, n_informative=10,
                               seed=42, **kwd):
        random_state = np.random.RandomState(seed=seed)
        X, y, coeff = make_regression(n_samples=n_instances * n_objects,
                                      n_features=n_features,
                                      n_informative=n_informative, coef=True,
                                      noise=noise,
                                      random_state=random_state)
        X = X.reshape(n_instances, n_objects, n_features)
        y = y.reshape(n_instances, n_objects)
        rankings = scores_to_rankings(y)
        return X, rankings

    def make_gp_transitive(self, n_instances=1000, n_objects=5, noise=0.0,
                           n_features=100, kernel_params=None, seed=42, **kwd):
        """Creates a nonlinear object ranking problem by sampling from a
        Gaussian process as the latent utility function.
        Note that this function needs to compute a kernel matrix of size
        (n_instances * n_objects) ** 2, which could allocate a large chunk of the
        memory."""
        random_state = np.random.RandomState(seed=seed)

        if kernel_params is None:
            kernel_params = dict()
        n_total = n_instances * n_objects
        X = random_state.rand(n_total, n_features)
        L = np.linalg.cholesky(Matern(**kernel_params)(X))
        f = (L.dot(random_state.randn(n_total)) +
             random_state.normal(scale=noise, size=n_total))
        X = X.reshape(n_instances, n_objects, n_features)
        f = f.reshape(n_instances, n_objects)
        rankings = scores_to_rankings(f)

        return X, rankings

    def make_gp_non_transitive(self, n_instances=1000, n_objects=5,
                               n_features=100, center_box=(-10.0, 10.0),
                               cluster_std=2.0, seed=42, **kwd):
        n_samples = n_instances * n_objects
        random_state = np.random.RandomState(seed=seed)
        x, y = make_blobs(n_samples=n_samples, centers=n_objects,
                          n_features=n_features, cluster_std=cluster_std,
                          center_box=center_box, random_state=random_state,
                          shuffle=True)
        y = np.array([y])
        samples = np.append(x, y.T, axis=1)
        samples = samples[samples[:, n_features].argsort()]
        pairwise_prob = create_pairwise_prob_matrix(n_objects)
        X = []
        rankings = []
        for inst in range(n_instances):
            feature = np.array([samples[inst + i * n_instances, 0:-1] for i in
                                range(n_objects)])
            matrix = np.random.binomial(1, pairwise_prob)
            objects = np.arange(n_objects)
            ranking = np.array(quicksort(objects, matrix))
            ordering = np.array(
                [np.where(obj == ranking)[0][0] for obj in objects])
            X.append(feature)
            rankings.append(ordering)
        X = np.array(X)
        rankings = np.array(rankings)
        return X, rankings

    def make_intransitive_medoids(self, n_instances=100, n_objects=5,
                                  n_features=100, seed=42, **kwd):
        random_state = np.random.RandomState(seed=seed)
        X = random_state.uniform(size=(n_instances, n_objects, n_features))
        rankings = np.empty((n_instances, n_objects))
        for i in range(n_instances):
            D = squareform(pdist(X[i], metric='euclidean'))
            sum_dist = D.mean(axis=0)
            medoid = np.argmin(sum_dist)
            ordering = np.argsort(D[medoid])
            ranking = np.argsort(ordering)
            rankings[i] = ranking
        X = np.array(X)
        rankings = np.array(rankings)
        return X, rankings

    def make_hv_dataset(self, n_instances=1000, n_objects=5, n_features=5,
                        seed=42, **kwd):
        random_state = np.random.RandomState(seed=seed)
        X = random_state.randn(n_instances, n_objects, n_features)
        # Normalize to unit circle and fold to lower quadrant
        X = -np.abs(X / np.sqrt(np.power(X, 2).sum(axis=2))[..., None])
        Y = np.empty((n_instances, n_objects), dtype=int)
        reference = np.zeros(n_features)
        for i, x in enumerate(X):
            hv = hypervolume(x)
            cont = hv.contributions(reference)
            Y[i] = np.argsort(cont)[::-1].argsort()

        return X, Y
