import logging

from keras.optimizers import SGD
from keras.regularizers import l2

from csrank.core.feta_network import FETANetwork
from csrank.losses import hinged_rank_loss
from .object_ranker import ObjectRanker

__all__ = ['FETAObjectRanker']


class FETAObjectRanker(FETANetwork, ObjectRanker):
    def __init__(self, n_objects, n_object_features, n_hidden=2, n_units=8, add_zeroth_order_model=False,
                 max_number_of_objects=5, num_subsample=5, loss_function=hinged_rank_loss, batch_normalization=False,
                 kernel_regularizer=l2(l=1e-4), kernel_initializer='lecun_normal', activation='selu',
                 optimizer=SGD(lr=1e-4, nesterov=True, momentum=0.9), metrics=None, batch_size=256, random_state=None,
                 **kwargs):
        """
            Create a FETA-network architecture for object ranking. The first-evaluate-then-aggregate approach
            approximates the context-dependent utility function using the first-order utility function
            :math:`U_1 \colon \mathcal{X} \\times \mathcal{X} \\rightarrow [0,1]` and zeroth-order utility
            function :math:`U_0 \colon \mathcal{X} \\rightarrow [0,1]`.
            The scores each object :math:`x` using a context-dependent utility function :math:`U (x, C_i)`:

            .. math::
                 U(x_i, C_i) = U_0(x_i) + \\frac{1}{n-1} \sum_{x_j \in Q \\setminus \{x_i\}} U_1(x_i , x_j) \, .

            Training and prediction complexity is quadratic in the number of objects.
            The ranking for the given query set :math:`Q` is defined as:

            .. math::
                ρ(Q)  = \operatorname{argsort}_{x_i \in Q}  \; U (x_i, C_i)

            Parameters
            ----------
            n_objects : int
                Number of objects to be ranked
            n_object_features : int
                Dimensionality of the feature space of each object
            n_hidden : int
                Number of hidden layers
            n_units : int
                Number of hidden units in each layer
            add_zeroth_order_model : bool
                True if the model should include a latent utility function
            max_number_of_objects : int
                The maximum number of objects to train from
            num_subsample : int
                Number of objects to subsample to
            loss_function : function
                Differentiable loss function for the score vector
            batch_normalization : bool
                Whether to use batch normalization in the hidden layers
            kernel_regularizer : function
                Regularizer to use in the hidden units
            kernel_initializer : function or string
                Initialization function for the weights of each hidden layer
            activation : string or function
                Activation function to use in the hidden units
            optimizer : string or function
                Stochastic gradient optimizer
            metrics : list
                List of evaluation metrics (can be non-differentiable)
            batch_size : int
                Batch size to use for training
            random_state : int or object
                Numpy random state
            **kwargs
                Keyword arguments for the hidden units
        """
        super().__init__(n_objects=n_objects, n_object_features=n_object_features, n_hidden=n_hidden, n_units=n_units,
                         add_zeroth_order_model=add_zeroth_order_model, max_number_of_objects=max_number_of_objects,
                         num_subsample=num_subsample, loss_function=loss_function,
                         batch_normalization=batch_normalization, kernel_regularizer=kernel_regularizer,
                         kernel_initializer=kernel_initializer, activation=activation, optimizer=optimizer,
                         metrics=metrics, batch_size=batch_size, random_state=random_state, **kwargs)
        self.logger = logging.getLogger(FETAObjectRanker.__name__)

    def construct_model(self):
        return super().construct_model()

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

    def clear_memory(self, **kwargs):
        self.logger.info("Clearing memory")
        super().clear_memory(**kwargs)

    def set_tunable_parameters(self, n_hidden=32, n_units=2, reg_strength=1e-4, learning_rate=1e-3, batch_size=128,
                               **point):
        super().set_tunable_parameters(n_hidden=n_hidden, n_units=n_units, reg_strength=reg_strength,
                                       learning_rate=learning_rate, batch_size=batch_size, **point)
