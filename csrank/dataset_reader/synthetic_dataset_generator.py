import logging

from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state

from csrank.dataset_reader.util import standardize_features
from .dataset_reader import DatasetReader


class SyntheticDatasetGenerator(DatasetReader):
    def __init__(self, learning_problem, n_train_instances=10000, n_test_instances=10000, random_state=None,
                 standardize=True, **kwargs):
        super(SyntheticDatasetGenerator, self).__init__(
            learning_problem=learning_problem, dataset_folder=None, **kwargs)
        self.random_state = check_random_state(random_state)
        self.dataset_function = None
        self.kwargs = kwargs
        self.n_test_instances = n_test_instances
        self.n_train_instances = n_train_instances
        self.logger = logging.getLogger(SyntheticDatasetGenerator.__name__)
        self.standardize = standardize
        self.logger.info("Key word arguments {}".format(kwargs))

    def __load_dataset__(self):
        pass

    def splitter(self, iter):
        for i in iter:
            seed = self.random_state.randint(2 ** 32, dtype='uint32')
            total_instances = self.n_test_instances + self.n_train_instances
            self.kwargs['n_instances'] = total_instances
            X, Y = self.dataset_function(**self.kwargs, seed=seed)
            x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=self.random_state,
                                                                test_size=self.n_test_instances)
            if self.standardize:
                x_train, x_test = standardize_features(x_train, x_test)
            yield x_train, y_train, x_test, y_test

    def get_dataset_dictionaries(self, lengths=[5, 6]):
        x_train = dict()
        y_train = dict()
        x_test = dict()
        y_test = dict()
        for n_obj in lengths:
            self.kwargs['n_objects'] = n_obj

            seed = self.random_state.randint(2 ** 32, dtype='uint32')
            total_instances = self.n_test_instances + self.n_train_instances
            self.kwargs['n_instances'] = total_instances
            X, Y = self.dataset_function(**self.kwargs, seed=seed)
            x_1, x_2, y_1, y_2 = train_test_split(X, Y, random_state=self.random_state, test_size=self.n_test_instances)
            if self.standardize:
                x_1, x_2 = standardize_features(x_1, x_2)
            x_train[n_obj], x_test[n_obj], y_train[n_obj], y_test[n_obj] = x_1, x_2, y_1, y_2
        self.logger.info('Done')
        return x_train, y_train, x_test, y_test

    def get_single_train_test_split(self):
        seed = self.random_state.randint(2 ** 32, dtype='uint32')
        total_instances = self.n_test_instances + self.n_train_instances
        self.kwargs['n_instances'] = total_instances
        self.X, self.Y = self.dataset_function(**self.kwargs, seed=seed)
        self.__check_dataset_validity__()
        x_train, x_test, y_train, y_test = train_test_split(self.X, self.Y, random_state=self.random_state,
                                                            test_size=self.n_test_instances)
        if self.standardize:
            x_train, x_test = standardize_features(x_train, x_test)
        self.logger.info('Done')

        return x_train, y_train, x_test, y_test


class SyntheticIterator(object):

    def __init__(self, dataset_function, **params):
        """
        Infinite iterator over a synthetic dataset generator.

        Parameters
        ----------
        dataset_function : callable
            Returns a tuple (inputs, targets) when called
        params : dict
            Parameters to be passed to `dataset_function` when called
        """
        self.params = params
        self.func = dataset_function

    def __iter__(self):
        return self

    def __next__(self):
        return self.func(**self.params)

    def __len__(self):
        """Return a constant to allow for steps per epoch."""
        return 100
