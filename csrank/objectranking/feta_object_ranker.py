import logging

from keras.optimizers import SGD
from keras.regularizers import l2

from csrank.core.feta_network import FETANetwork
from csrank.losses import hinged_rank_loss
from .object_ranker import ObjectRanker

__all__ = ["FETAObjectRanker"]


class FETAObjectRanker(ObjectRanker, FETANetwork):
    def __init__(
        self,
        n_hidden=2,
        n_units=8,
        add_zeroth_order_model=False,
        max_number_of_objects=5,
        num_subsample=5,
        loss_function=hinged_rank_loss,
        batch_normalization=False,
        kernel_regularizer=l2,
        kernel_initializer="lecun_normal",
        activation="selu",
        optimizer=SGD,
        metrics=(),
        batch_size=256,
        random_state=None,
        **kwargs,
    ):
        """
            Create a FETA-network architecture for object ranking. The first-evaluate-then-aggregate approach
            approximates the context-dependent utility function using the first-order utility function
            :math:`U_1 \\colon \\mathcal{X} \\times \\mathcal{X} \\rightarrow [0,1]` and zeroth-order utility
            function :math:`U_0 \\colon \\mathcal{X} \\rightarrow [0,1]`.
            The scores each object :math:`x` using a context-dependent utility function :math:`U (x, C_i)`:

            .. math::
                 U(x_i, C_i) = U_0(x_i) + \\frac{1}{n-1} \\sum_{x_j \\in Q \\setminus \\{x_i\\}} U_1(x_i , x_j) \\, .

            Training and prediction complexity is quadratic in the number of objects.
            The ranking for the given query set :math:`Q` is defined as:

            .. math::
                ρ(Q)  = \\operatorname{argsort}_{x_i \\in Q}  \\; U (x_i, C_i)

            Parameters
            ----------
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
            kernel_regularizer : uninitialized keras regularizer
                Regularizer to use in the hidden units
            kernel_initializer : function or string
                Initialization function for the weights of each hidden layer
            activation : string or function
                Activation function to use in the hidden units
            optimizer: Class
                Uninitialized optimizer class following the keras optimizer interface.
            optimizer__{kwarg}
                Arguments to be passed to the optimizer on initialization, such as optimizer__lr.
            metrics : list
                List of evaluation metrics (can be non-differentiable)
            batch_size : int
                Batch size to use for training
            random_state : int or object
                Numpy random state
            **kwargs
                Keyword arguments for the hidden units
        """
        super().__init__(
            n_hidden=n_hidden,
            n_units=n_units,
            add_zeroth_order_model=add_zeroth_order_model,
            max_number_of_objects=max_number_of_objects,
            num_subsample=num_subsample,
            loss_function=loss_function,
            batch_normalization=batch_normalization,
            kernel_regularizer=kernel_regularizer,
            kernel_initializer=kernel_initializer,
            activation=activation,
            optimizer=optimizer,
            metrics=metrics,
            batch_size=batch_size,
            random_state=random_state,
            **kwargs,
        )
        self.logger = logging.getLogger(FETAObjectRanker.__name__)
