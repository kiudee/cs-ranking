import numpy as np
from sklearn.utils import check_random_state

from csrank.constants import CHOICE_FUNCTION
from ..synthetic_dataset_generator import SyntheticDatasetGenerator


class ChoiceDatasetGenerator(SyntheticDatasetGenerator):
    def __init__(self, dataset_type="pareto", **kwargs):
        super(ChoiceDatasetGenerator, self).__init__(
            learning_problem=CHOICE_FUNCTION, **kwargs
        )
        dataset_function_options = {
            "linear": self.make_latent_linear_choices,
            "pareto": self.make_globular_pareto_choices,
        }
        if dataset_type not in dataset_function_options:
            raise ValueError(
                f"dataset_type must be one of {set(dataset_function_options.keys())}"
            )
        self.dataset_function = dataset_function_options[dataset_type]

    def make_globular_pareto_choices(
        self,
        n_instances=10000,
        n_features=2,
        n_objects=10,
        seed=42,
        cluster_spread=1.0,
        **kwargs,
    ):
        def pareto_front(X, signs=None):
            n_points, n_attributes = X.shape
            if signs is None:
                signs = -np.ones(n_attributes)
            pareto = np.ones(n_points, dtype=bool)
            for i, attr in enumerate(X):
                pareto[i] = np.all(
                    np.any((X * signs[None, :]) <= (attr * signs), axis=1)
                )
            return pareto

        def sample_from_unit_ball(n_points, dimension, radius, random_state):
            """Sample points uniformly from a ball.

            The ball has radius `radius` and is centered at the origin.

            Parameters
            ----------
            n_points : int
                The number of points to sample.
            dimension : int
                The dimension of the space.
            radius : float
                The radius of the ball.
            random_state: np.random.RandomState
                A numpy random state.

            Returns
            -------
            numpy array of shape (n_points, dimension)
                A list of points sampled from the ball.
            """
            # Sample a random direction for each point
            directions = random_state.randn(n_points, dimension)
            # Normalize each direction vector to have length 1 (euclidean
            # norm).
            directions /= np.linalg.norm(directions, axis=1, ord=2)[:, None]

            # Sample a length (as a fraction of the radius) uniformly for each
            # point.
            u = random_state.uniform(size=n_points)[:, None]
            lengths = u * radius

            return directions * lengths

        def sample_pareto_from_isometric_normal(
            n_points, dimension, center, random_state
        ):
            """Generate a Pareto problem from random objects.

            Objects are drawn from a d-dimensional isometric normal
            distribution.

            This should be the easiest possible Pareto-problem, since the model can learn
            a latent-utility which scores how likely a point is on the front (independent
            of the other points).

            Parameters
            ----------
            n_points : int
                The number of points to sample.
            dimension : int
                The dimension of the space.
            center : scalar or numpy array
                An offset that will be added to every point.
            random_state: np.random.RandomState
                A numpy random state.

            Returns
            -------
            X: numpy array of shape (n_points, dimension)
                A list of points sampled from the d-dimensional isometric
                normal distribution.
            Y. numpy array of shape n_points
                A binary flag array indicating whether or not the corresponding
                point is part of the Pareto front.
            """
            X = random_state.randn(n_points, dimension)
            Y = pareto_front(X)
            return X + center, Y

        rand = check_random_state(seed)
        X = np.empty((n_instances, n_objects, n_features))
        Y = np.empty((n_instances, n_objects), dtype=int)
        for i in range(n_instances):
            center = sample_from_unit_ball(
                n_points=1,
                dimension=n_features,
                radius=cluster_spread,
                random_state=rand,
            )
            x, y = sample_pareto_from_isometric_normal(
                n_points=n_objects,
                dimension=n_features,
                center=center,
                random_state=rand,
            )
            X[i] = x
            Y[i] = y
        return X, Y

    def make_latent_linear_choices(
        self,
        n_instances=10000,
        n_features=2,
        n_objects=6,
        n_rep_units=5,
        threshold=0.0,
        seed=42,
        **kwargs,
    ):
        rand = check_random_state(seed)
        ranw = check_random_state(rand.randint(2 ** 32, dtype="uint32"))
        X = rand.uniform(-1, 1, size=(n_instances, n_objects, n_features))
        W_rep = ranw.randn(n_features, n_rep_units)
        rep = X.dot(W_rep).mean(axis=-2)
        w_join = ranw.randn(n_features + n_rep_units)
        joint_matrix = np.empty(
            (n_objects, n_instances, n_features + n_rep_units), dtype=np.float32
        )
        for i in range(n_objects):
            joint_matrix[i] = np.concatenate((X[:, i], rep), axis=-1)
        scores = joint_matrix.dot(w_join)
        Y = scores > threshold
        Y = Y.astype(int)
        return X, Y.T
