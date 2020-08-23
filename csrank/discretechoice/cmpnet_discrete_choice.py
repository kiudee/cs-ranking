import logging

from keras.optimizers import SGD
from keras.regularizers import l2

from csrank.choicefunction.util import generate_complete_pairwise_dataset
from csrank.core.cmpnet_core import CmpNetCore
from csrank.discretechoice.discrete_choice import DiscreteObjectChooser


class CmpNetDiscreteChoiceFunction(DiscreteObjectChooser, CmpNetCore):
    def __init__(
        self,
        n_hidden=2,
        n_units=8,
        loss_function="binary_crossentropy",
        batch_normalization=True,
        kernel_regularizer=l2,
        kernel_initializer="lecun_normal",
        activation="relu",
        optimizer=SGD,
        metrics=("binary_accuracy",),
        batch_size=256,
        random_state=None,
        **kwargs,
    ):
        """
            Create an instance of the :class:`CmpNetCore` architecture for learning a discrete choice function.
            CmpNet breaks the preferences in form of rankings into pairwise comparisons and learns a pairwise model for
            the each pair of object in the underlying set. For prediction list of objects is converted in pair of
            objects and the pairwise predicate is evaluated using them. The outputs of the network for each pair of
            objects :math:`U(x_1,x_2), U(x_2,x_1)` are evaluated.
            :math:`U(x_1,x_2)` is a measure of how favorable it is to choose :math:`x_1` over :math:`x_2`.
            The utility score of object :math:`x_i` in query set :math:`Q = \\{ x_1 , \\ldots , x_n \\}` is evaluated as:

            .. math::

                U(x_i) = \\left\\{ \\frac{1}{n-1} \\sum_{j \\in [n] \\setminus \\{i\\}} U_1(x_i , x_j)\\right\\}

            The discrete choice for the given query set :math:`Q` is defined as:

            .. math::

                dc(Q) := \\operatorname{argmax}_{i \\in [n]}  \\; U(x_i)

            Parameters
            ----------
            n_hidden : int
                Number of hidden layers used in the scoring network
            n_units : int
                Number of hidden units in each layer of the scoring network
            loss_function : function or string
                Loss function to be used for the binary decision task of the pairwise comparisons
            batch_normalization : bool
                Whether to use batch normalization in each hidden layer
            kernel_regularizer : uninitialized keras regularizer
                Regularizer function applied to all the hidden weight matrices
            kernel_initializer : function or string
                Initialization function for the weights of each hidden layer
            activation : function or string
                Type of activation function to use in each hidden layer
            optimizer: Class
                Uninitialized optimizer class following the keras optimizer interface.
            optimizer__{kwarg}
                Arguments to be passed to the optimizer on initialization, such as optimizer__lr.
            metrics : list
                List of metrics to evaluate during training (can be non-differentiable)
            batch_size : int
                Batch size to use during training
            random_state : int, RandomState instance or None
                Seed of the pseudorandom generator or a RandomState instance
            **kwargs
                Keyword arguments for the algorithms

            References
            ----------
                [1] Leonardo Rigutini, Tiziano Papini, Marco Maggini, and Franco Scarselli. 2011. SortNet: Learning to Rank by a Neural Preference Function. IEEE Trans. Neural Networks 22, 9 (2011), 1368–1380. https://doi.org/10.1109/TNN.2011.2160875
        """
        super().__init__(
            n_hidden=n_hidden,
            n_units=n_units,
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
        self.logger = logging.getLogger(CmpNetDiscreteChoiceFunction.__name__)
        self.logger.info("Initializing network")

    def _convert_instances_(self, X, Y):
        self.logger.debug("Creating the Dataset")
        x1, x2, garbage, y_double, garbage = generate_complete_pairwise_dataset(X, Y)
        del garbage
        self.logger.debug("Finished the Dataset instances {}".format(x1.shape[0]))
        return x1, x2, y_double
