from scipy.spatial.distance import squareform, pdist
from sklearn.datasets import make_regression
from sklearn.utils import check_random_state

from csrank.constants import DISCRETE_CHOICE
from csrank.dataset_reader import SyntheticDatasetGenerator
import numpy as np


class DiscreteChoiceDatasetGenerator(SyntheticDatasetGenerator):
    def __init__(self, dataset_type='medoid', **kwargs):
        super(DiscreteChoiceDatasetGenerator, self).__init__(
            learning_problem=DISCRETE_CHOICE, **kwargs)
        dataset_function_options = {'linear': self.make_linear_transitive,
                                    'medoid': self.make_intransitive_medoids}
        if dataset_type not in dataset_function_options.keys():
            dataset_type = "medoid"
        self.dataset_function = dataset_function_options[dataset_type]

    def make_linear_transitive(self, n_instances=1000, n_objects=5, noise=0.0,
                               n_features=100, n_informative=10,
                               seed=42, **kwd):
        random_state = check_random_state(seed=seed)
        X, y, coeff = make_regression(n_samples=n_instances * n_objects,
                                      n_features=n_features,
                                      n_informative=n_informative, coef=True,
                                      noise=noise,
                                      random_state=random_state)
        X = X.reshape(n_instances, n_objects, n_features)
        y = y.reshape(n_instances, n_objects)
        Y = y.argmax(axis=1)
        return X, Y

    def make_intransitive_medoids(self, n_instances=100, n_objects=5,
                                  n_features=100, seed=42, **kwd):
        random_state = check_random_state(seed=seed)
        X = random_state.uniform(size=(n_instances, n_objects, n_features))
        Y = np.empty((n_instances))
        for i in range(n_instances):
            D = squareform(pdist(X[i], metric='euclidean'))
            sum_dist = D.mean(axis=0)
            medoid = np.argmin(sum_dist)
            Y[i] = medoid
        X = np.array(X)
        Y = np.array(Y)
        return X, Y

    def get_single_train_test_split(self):
        return super(DiscreteChoiceDatasetGenerator, self).get_single_train_test_split()

    def get_train_test_datasets(self, n_datasets=5):
        return super(DiscreteChoiceDatasetGenerator, self).get_train_test_datasets(n_datasets=n_datasets)
