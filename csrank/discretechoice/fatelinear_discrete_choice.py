import logging

from keras.losses import categorical_hinge

from csrank.core.fate_linear import FATELinearCore
from csrank.discretechoice.discrete_choice import DiscreteObjectChooser

logger = logging.getLogger(__name__)


class FATELinearDiscreteChoiceFunction(DiscreteObjectChooser, FATELinearCore):
    def __init__(
        self,
        n_hidden_set_units=32,
        loss_function=categorical_hinge,
        learning_rate=1e-3,
        batch_size=256,
        random_state=None,
        **kwargs,
    ):
        """
            Create a FATELinear-network architecture for leaning discrete choice function. The first-aggregate-then-evaluate
            approach learns an embedding of each object and then aggregates that into a context representation
            :math:`\\mu_{C(x)}` and then scores each object :math:`x` using a generalized utility function
            :math:`U (x, \\mu_{C(x)})`.
            To make it computationally efficient we take the the context :math:`C(x)` as query set :math:`Q`.
            The context-representation is evaluated as:

            .. math::
                \\mu_{C(x)} = \\frac{1}{\\lvert C(x) \\lvert} \\sum_{y \\in C(x)} \\phi(y)

            where :math:`\\phi \\colon \\mathcal{X} \\to \\mathcal{Z}` maps each object :math:`y` to an
            :math:`m`-dimensional embedding space :math:`\\mathcal{Z} \\subseteq \\mathbb{R}^m`.
            Training complexity is quadratic in the number of objects and prediction complexity is only linear.
            The discrete choice for the given query set :math:`Q` is defined as:

            .. math::

                dc(Q) := \\operatorname{argmax}_{x \\in Q}  \\;  U (x, \\mu_{C(x)})

            Parameters
            ----------
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
        super().__init__(
            n_hidden_set_units=n_hidden_set_units,
            learning_rate=learning_rate,
            batch_size=batch_size,
            loss_function=loss_function,
            random_state=random_state,
            **kwargs,
        )
