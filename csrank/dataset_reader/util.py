import collections


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
