from itertools import permutations
import logging

from keras import Input
from keras import Model
from keras.layers import concatenate
from keras.layers import Dense
from keras.optimizers import SGD
from keras.regularizers import l2
import numpy as np
from sklearn.utils import check_random_state

from csrank.layers import NormalizedDense
from csrank.learner import Learner

logger = logging.getLogger(__name__)


class CmpNetCore(Learner):
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
        self.batch_normalization = batch_normalization
        self.activation = activation

        self.batch_size = batch_size

        self.metrics = metrics
        self.kernel_regularizer = kernel_regularizer
        self.kernel_initializer = kernel_initializer
        self.loss_function = loss_function

        self.optimizer = optimizer

        self.n_hidden = n_hidden
        self.n_units = n_units
        self.random_state = random_state
        self._store_kwargs(
            kwargs, {"kernel_regularizer__", "optimizer__", "hidden_dense_layer__"}
        )

    def _construct_layers(self):

        self.output_node = Dense(
            1, activation="sigmoid", kernel_regularizer=self.kernel_regularizer_
        )

        self.x1 = Input(shape=(self.n_object_features_fit_,))
        self.x2 = Input(shape=(self.n_object_features_fit_,))
        hidden_dense_kwargs = {
            "kernel_regularizer": self.kernel_regularizer_,
            "kernel_initializer": self.kernel_initializer,
            "activation": self.activation,
        }
        hidden_dense_kwargs.update(self._get_prefix_attributes("hidden_dense_layer__"))
        if self.batch_normalization:
            self.hidden_layers = [
                NormalizedDense(
                    self.n_units, name="hidden_{}".format(x), **hidden_dense_kwargs
                )
                for x in range(self.n_hidden)
            ]
        else:
            self.hidden_layers = [
                Dense(self.n_units, name="hidden_{}".format(x), **hidden_dense_kwargs)
                for x in range(self.n_hidden)
            ]
        assert len(self.hidden_layers) == self.n_hidden

    def _convert_instances_(self, X, Y):
        raise NotImplementedError

    def construct_model(self):
        """
            Construct the CmpNet which is used to approximate the :math:`U_1(x_i,x_j)`. For each pair of objects in
            :math:`x_i, x_j \\in Q` we construct two sub-networks with weight sharing in all hidden layers.
            The output of these networks are connected to two sigmoid units that produces the outputs of the network,
            i.e., :math:`U(x_1,x_2), U(x_2,x_1)` for each pair of objects are evaluated. :math:`U(x_1,x_2)` is a measure
            of how favorable it is to choose :math:`x_1` over :math:`x_2`.

            Returns
            -------
            model: keras :class:`Model`
                Neural network to learn the CmpNet utility score
        """
        x1x2 = concatenate([self.x1, self.x2])
        x2x1 = concatenate([self.x2, self.x1])
        logger.debug("Creating the model")
        for hidden in self.hidden_layers:
            x1x2 = hidden(x1x2)
            x2x1 = hidden(x2x1)
        merged_left = concatenate([x1x2, x2x1])
        merged_right = concatenate([x2x1, x1x2])
        N_g = self.output_node(merged_left)
        N_l = self.output_node(merged_right)
        merged_output = concatenate([N_g, N_l])
        model = Model(inputs=[self.x1, self.x2], outputs=merged_output)
        model.compile(
            loss=self.loss_function,
            optimizer=self.optimizer_,
            metrics=list(self.metrics),
        )
        return model

    def _pre_fit(self):
        super()._pre_fit()
        self.random_state_ = check_random_state(self.random_state)
        self._initialize_optimizer()
        self._initialize_regularizer()

    def fit(
        self, X, Y, epochs=10, callbacks=None, validation_split=0.1, verbose=0, **kwd
    ):
        """
            Fit a generic preference learning CmptNet on the provided set of queries X and preferences Y of those
            objects. The provided queries and corresponding preferences are of a fixed size (numpy arrays).
            For learning this network the binary cross entropy loss function for a pair of objects
            :math:`x_i, x_j \\in Q` is defined as:

            .. math::

                C_{ij} =  -\\tilde{P_{ij}}(0)\\cdot \\log(U(x_i,x_j)) - \\tilde{P_{ij}}(1) \\cdot \\log(U(x_j,x_i)) \\ ,

            where :math:`\\tilde{P_{ij}}` is ground truth probability of the preference of :math:`x_i` over :math:`x_j`.
            :math:`\\tilde{P_{ij}} = (1,0)` if :math:`x_i \\succ x_j` else :math:`\\tilde{P_{ij}} = (0,1)`.

            Parameters
            ----------
            X : numpy array
                (n_instances, n_objects, n_features)
                Feature vectors of the objects
            Y : numpy array
                (n_instances, n_objects)
                Preferences in form of Orderings or Choices for given n_objects
            epochs : int
                Number of epochs to run if training for a fixed query size
            callbacks : list
                List of callbacks to be called during optimization
            validation_split : float (range : [0,1])
                Percentage of instances to split off to validate on
            verbose : bool
                Print verbose information
            **kwd :
                Keyword arguments for the fit function
        """
        self._pre_fit()
        _n_instances, self.n_objects_fit_, self.n_object_features_fit_ = X.shape
        x1, x2, y_double = self._convert_instances_(X, Y)

        logger.debug("Instances created {}".format(x1.shape[0]))
        self._construct_layers()
        self.model_ = self.construct_model()

        logger.debug("Finished Creating the model, now fitting started")
        self.model_.fit(
            [x1, x2],
            y_double,
            batch_size=self.batch_size,
            epochs=epochs,
            callbacks=callbacks,
            validation_split=validation_split,
            verbose=verbose,
            **kwd,
        )
        logger.debug("Fitting Complete")

    def predict_pair(self, a, b, **kwargs):
        return self.model_.predict([a, b], **kwargs)

    def _predict_scores_fixed(self, X, **kwargs):
        n_instances, n_objects, n_features = X.shape
        logger.info("Test Set instances {} objects {} features {}".format(*X.shape))
        n2 = n_objects * (n_objects - 1)
        pairs = np.empty((n2, 2, n_features))
        scores = np.empty((n_instances, n_objects))
        for n in range(n_instances):
            for k, (i, j) in enumerate(permutations(range(n_objects), 2)):
                pairs[k] = (X[n, i], X[n, j])
            result = self.predict_pair(pairs[:, 0], pairs[:, 1], **kwargs)[:, 0]
            scores[n] = result.reshape(n_objects, n_objects - 1).mean(axis=1)
            del result
        del pairs
        logger.info("Done predicting scores")

        return scores
