import logging

from keras.optimizers import SGD
from keras.regularizers import l2

from csrank.losses import hinged_rank_loss
from .feta_object_ranker import FETAObjectRanker


class FETALinearObjectRanker(FETAObjectRanker):
    def __init__(self, n_object_features, n_objects, loss_function=hinged_rank_loss,
                 learning_rate=5e-3, batch_size=256, random_state=None, **kwargs):
        """
            Create a FETA-network architecture for learning discrete choice function. The first-evaluate-then-aggregate
            approach approximates the context-dependent utility function using the first-order utility
            function :math:`U_1 \colon \mathcal{X} \\times \mathcal{X} \\rightarrow [0,1]` and zeroth-order utility
            function  :math:`U_0 \colon \mathcal{X} \\rightarrow [0,1]`. The scores each object :math:`x` using a
            context-dependent utility function :math:`U (x, C_i)`:

            .. math::
                 U(x_i, C_i) = U_0(x_i) + \\frac{1}{n-1} \sum_{x_j \in Q \\setminus \{x_i\}} U_1(x_i , x_j) \, .

            Training and prediction complexity is quadratic in the number of objects.
            The choice set is defined as:

            .. math::

                c(Q) = \{ x_i \in Q \lvert \, U (x_i, C_i) > t \}

            Parameters
            ----------
            n_object_features : int
                Dimensionality of the feature space of each object
            n_objects : int
                Number of objects in each choice set
            loss_function : function
                Differentiable loss function for the score vector
            learning_rate: float >= 0.
                Learning rate for the optimizer
            batch_size : int
                Batch size to use for training
            random_state : int or object
                Numpy random state
            **kwargs
                Keyword arguments for the @FATENetwork
        """
        super().__init__(n_objects=n_objects, n_object_features=n_object_features, max_number_of_objects=n_objects,
                         n_hidden=1, n_units=1, add_zeroth_order_model=True, loss_function=loss_function,
                         batch_normalization=False, activation='selu', kernel_initializer='lecun_normal',
                         kernel_regularizer=l2(l=1e-4), optimizer=SGD(lr=learning_rate, nesterov=True, momentum=0.9),
                         batch_size=batch_size, metrics=None, random_state=random_state, **kwargs)
        self.logger = logging.getLogger(FETALinearObjectRanker.__name__)

    def fit(self, X, Y, epochs=10, callbacks=None, validation_split=0.1, verbose=0, **kwd):
        super().fit(X=X, Y=Y, epochs=epochs, callbacks=callbacks, validation_split=validation_split, verbose=verbose,
                    **kwd)

    def _predict_scores_fixed(self, X, **kwargs):
        return super()._predict_scores_fixed(X, **kwargs)

    def predict_scores(self, X, **kwargs):
        return super().predict_scores(X, **kwargs)

    def predict_for_scores(self, scores, **kwargs):
        return super().predict_for_scores(scores, **kwargs)

    def predict(self, X, **kwargs):
        return super().predict(X, **kwargs)

    def set_tunable_parameters(self, learning_rate=1e-3, reg_strength=1e-4, batch_size=128, **point):
        super().set_tunable_parameters(n_hidden=1, n_units=1, reg_strength=reg_strength, learning_rate=learning_rate,
                                       batch_size=batch_size, **point)
