

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
        while True:
            yield self.func(**self.params)
