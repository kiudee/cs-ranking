from sklearn.utils import check_random_state

from .dataset_reader import DatasetReader


class SyntheticDatasetGenerator(DatasetReader):
    def __init__(self, learning_problem, n_train_instances=10000, n_test_instances=10000, random_state=None, **kwargs):
        super(SyntheticDatasetGenerator, self).__init__(
            learning_problem=learning_problem, dataset_folder=None, **kwargs)
        self.random_state = check_random_state(random_state)
        self.dataset_function = None
        self.kwargs = kwargs
        self.n_test_instances = n_test_instances
        self.n_train_instances = n_train_instances
        self.dr_logger.info("Key word arguments {}".format(kwargs))

    def __load_dataset__(self):
        pass

    def splitter(self, iter):
        for i in iter:
            seed = self.random_state.randint(2 ** 32, dtype='uint32')
            self.kwargs['n_instances'] = self.n_train_instances
            X_train, Y_train = self.dataset_function(**self.kwargs, seed=seed)

            self.kwargs['n_instances'] = self.n_test_instances
            seed = self.random_state.randint(2 ** 32, dtype='uint32')
            X_test, Y_test = self.dataset_function(**self.kwargs, seed=seed)
        yield X_train, Y_train, X_test, Y_test

    def get_dataset_dictionaries(self, lengths=[5, 6]):
        X_train = dict()
        Y_train = dict()
        X_test = dict()
        Y_test = dict()
        for n_obj in lengths:
            self.kwargs['n_objects'] = n_obj

            seed = self.random_state.randint(2 ** 32, dtype='uint32')
            self.kwargs['n_instances'] = self.n_train_instances
            X_train[n_obj], Y_train[n_obj] = self.dataset_function(**self.kwargs, seed=seed)

            seed = self.random_state.randint(2 ** 32, dtype='uint32')
            self.kwargs['n_instances'] = self.n_test_instances
            X_test[n_obj], Y_test[n_obj] = self.dataset_function(**self.kwargs, seed=seed)
        return X_train, Y_train, X_test, Y_test

    def get_single_train_test_split(self):
        seed = self.random_state.randint(2 ** 32, dtype='uint32')
        self.kwargs['n_instances'] = self.n_train_instances
        self.X, self.Y = X_train, Y_train = self.dataset_function(**self.kwargs, seed=seed)
        self.__check_dataset_validity__()

        self.kwargs['n_instances'] = self.n_test_instances
        seed = self.random_state.randint(2 ** 32, dtype='uint32')
        self.X, self.Y = X_test, Y_test = self.dataset_function(**self.kwargs, seed=seed)
        self.__check_dataset_validity__()
        return X_train, Y_train, X_test, Y_test


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
