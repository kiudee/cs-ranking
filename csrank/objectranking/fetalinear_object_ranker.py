import logging

from csrank.core.feta_linear import FETALinearCore
from csrank.losses import hinged_rank_loss
from .object_ranker import ObjectRanker


class FETALinearObjectRanker(FETALinearCore, ObjectRanker):
    def __init__(self, n_object_features, n_objects, loss_function=hinged_rank_loss,
                 learning_rate=5e-3, batch_size=256, random_state=None, **kwargs):
        """
            Create a FATELinear-network architecture for leaning discrete choice function. The first-aggregate-then-evaluate
            approach learns an embedding of each object and then aggregates that into a context representation
            :math:`\\mu_{C(x)}` and then scores each object :math:`x` using a generalized utility function
            :math:`U (x, \\mu_{C(x)})`.
            To make it computationally efficient we take the the context :math:`C(x)` as query set :math:`Q`.
            The context-representation is evaluated as:

            .. math::
                \\mu_{C(x)} = \\frac{1}{\\lvert C(x) \\lvert} \\sum_{y \\in C(x)} \\phi(y)

            where :math:`\phi \colon \mathcal{X} \\to \mathcal{Z}` maps each object :math:`y` to an
            :math:`m`-dimensional embedding space :math:`\mathcal{Z} \subseteq \mathbb{R}^m`.
            Training complexity is quadratic in the number of objects and prediction complexity is only linear.
            The discrete choice for the given query set :math:`Q` is defined as:

            .. math::

                dc(Q) := \operatorname{argmax}_{x \in Q}  \;  U (x, \\mu_{C(x)})

            Parameters
            ----------
            n_object_features : int
                Dimensionality of the feature space of each object
            n_objects : int
                Number of objects in each choice set
            n_hidden_set_units : int
                Number of hidden set units.
            batch_size : int
                Batch size to use for training
            loss_function : function
                Differentiable loss function for the score vector
            random_state : int or object
                Numpy random state
            **kwargs
                Keyword arguments for the @FATENetwork
        """
        super().__init__(n_object_features=n_object_features, n_objects=n_objects, learning_rate=learning_rate,
                         batch_size=batch_size,
                         loss_function=loss_function, random_state=random_state, **kwargs)
        self.logger = logging.getLogger(FETALinearObjectRanker.__name__)

    def fit(self, X, Y, epochs=10, callbacks=None, validation_split=0.1, verbose=0, **kwd):
        """
            Fit an object ranking learning model on a provided set of queries.
            The provided queries are of a fixed size (numpy arrays).

            Parameters
            ----------
            X : numpy array
                (n_instances, n_objects, n_features)
                Feature vectors of the objects
            Y : numpy array
                (n_instances, n_objects)
                Rankings of the given objects
            epochs : int
                Number of epochs to run if training for a fixed query size
            callbacks : list
                List of callbacks to be called during optimization
            validation_split : float
                Percentage of instances to split off to validate on
            verbose : bool
                Print verbose information
            **kwd
                Keyword arguments for the fit function
        """
        super().fit(X, Y, epochs=epochs, callbacks=callbacks, validation_split=validation_split, verbose=verbose, **kwd)

    def _predict_scores_fixed(self, X, **kwargs):
        return super()._predict_scores_fixed(X, **kwargs)

    def predict_scores(self, X, **kwargs):
        return super().predict_scores(X, **kwargs)

    def predict_for_scores(self, scores, **kwargs):
        return ObjectRanker.predict_for_scores(self, scores, **kwargs)

    def predict(self, X, **kwargs):
        return super().predict(X, **kwargs)

    def set_tunable_parameters(self, learning_rate=1e-3, batch_size=128, epochs_drop=300,
                               drop=0.1, **point):
        super().set_tunable_parameters(learning_rate=learning_rate, batch_size=batch_size, epochs_drop=epochs_drop,
                                       drop=drop, **point)
