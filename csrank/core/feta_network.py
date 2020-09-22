from itertools import combinations
from itertools import permutations
import logging

from keras import backend as K
from keras import Input
from keras import Model
from keras.layers import add
from keras.layers import concatenate
from keras.layers import Dense
from keras.layers import Lambda
from keras.optimizers import SGD
from keras.regularizers import l2
import numpy as np
from sklearn.utils import check_random_state

from csrank.layers import NormalizedDense
from csrank.learner import Learner
from csrank.losses import hinged_rank_loss

logger = logging.getLogger(__name__)


class FETANetwork(Learner):
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
        self.random_state = random_state
        self.kernel_regularizer = kernel_regularizer
        self.kernel_initializer = kernel_initializer
        self.batch_normalization = batch_normalization
        self.activation = activation
        self.loss_function = loss_function
        self.metrics = metrics
        self.max_number_of_objects = max_number_of_objects
        self.num_subsample = num_subsample
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.add_zeroth_order_model = add_zeroth_order_model
        self.n_hidden = n_hidden
        self.n_units = n_units
        self._store_kwargs(
            kwargs, {"optimizer__", "kernel_regularizer__", "hidden_dense_layer__"}
        )

    @property
    def n_objects(self):
        if self.n_objects_fit_ > self.max_number_of_objects:
            return self.max_number_of_objects
        return self.n_objects_fit_

    def _construct_layers(self):
        self.input_layer = Input(
            shape=(self.n_objects_fit_, self.n_object_features_fit_)
        )
        # Todo: Variable sized input
        # X = Input(shape=(None, n_features))
        logger.info("n_hidden {}, n_units {}".format(self.n_hidden, self.n_units))
        hidden_dense_kwargs = {
            "kernel_regularizer": self.kernel_regularizer_,
            "kernel_initializer": self.kernel_initializer,
            "activation": self.activation,
        }
        hidden_dense_kwargs.update(self._get_prefix_attributes("hidden_dense_layer__"))
        if self.batch_normalization:
            if self.add_zeroth_order_model:
                self.hidden_layers_zeroth = [
                    NormalizedDense(
                        self.n_units,
                        name="hidden_zeroth_{}".format(x),
                        **hidden_dense_kwargs,
                    )
                    for x in range(self.n_hidden)
                ]
            self.hidden_layers = [
                NormalizedDense(
                    self.n_units, name="hidden_{}".format(x), **hidden_dense_kwargs
                )
                for x in range(self.n_hidden)
            ]
        else:
            if self.add_zeroth_order_model:
                self.hidden_layers_zeroth = [
                    Dense(
                        self.n_units,
                        name="hidden_zeroth_{}".format(x),
                        **hidden_dense_kwargs,
                    )
                    for x in range(self.n_hidden)
                ]
            self.hidden_layers = [
                Dense(self.n_units, name="hidden_{}".format(x), **hidden_dense_kwargs)
                for x in range(self.n_hidden)
            ]
        assert len(self.hidden_layers) == self.n_hidden
        self.output_node = Dense(
            1, activation="sigmoid", kernel_regularizer=self.kernel_regularizer_
        )
        if self.add_zeroth_order_model:
            self.output_node_zeroth = Dense(
                1, activation="sigmoid", kernel_regularizer=self.kernel_regularizer_
            )

    @property
    def zero_order_model(self):
        if not hasattr(self, "zero_order_model_"):
            if self.add_zeroth_order_model:
                logger.info("Creating zeroth model")
                inp = Input(shape=(self.n_object_features_fit_,))

                x = inp
                for hidden in self.hidden_layers_zeroth:
                    x = hidden(x)
                zeroth_output = self.output_node_zeroth(x)

                self.zero_order_model_ = Model(inputs=[inp], outputs=zeroth_output)
                logger.info("Done creating zeroth model")
            else:
                self.zero_order_model_ = None
        return self.zero_order_model_

    @property
    def pairwise_model(self):
        if not hasattr(self, "pairwise_model_"):
            logger.info("Creating pairwise model")
            x1 = Input(shape=(self.n_object_features_fit_,))
            x2 = Input(shape=(self.n_object_features_fit_,))

            x1x2 = concatenate([x1, x2])
            x2x1 = concatenate([x2, x1])

            for hidden in self.hidden_layers:
                x1x2 = hidden(x1x2)
                x2x1 = hidden(x2x1)

            merged_left = concatenate([x1x2, x2x1])
            merged_right = concatenate([x2x1, x1x2])

            n_g = self.output_node(merged_left)
            n_l = self.output_node(merged_right)

            merged_output = concatenate([n_g, n_l])
            self.pairwise_model_ = Model(inputs=[x1, x2], outputs=merged_output)
            logger.info("Done creating pairwise model")
        return self.pairwise_model_

    def _predict_pair(self, a, b, only_pairwise=False, **kwargs):
        # TODO: Is this working correctly?
        pairwise = self.pairwise_model.predict([a, b], **kwargs)
        if not only_pairwise and self.add_zeroth_order_model:
            utility_a = self.zero_order_model.predict([a])
            utility_b = self.zero_order_model.predict([b])
            return pairwise + (utility_a, utility_b)
        return pairwise

    def _predict_scores_using_pairs(self, X, **kwd):
        n_instances, n_objects, n_features = X.shape
        n2 = n_objects * (n_objects - 1)
        pairs = np.empty((n2, 2, n_features))
        scores = np.zeros((n_instances, n_objects))
        for n in range(n_instances):
            for k, (i, j) in enumerate(permutations(range(n_objects), 2)):
                pairs[k] = (X[n, i], X[n, j])
            result = self._predict_pair(
                pairs[:, 0], pairs[:, 1], only_pairwise=True, **kwd
            )[:, 0]
            scores[n] += result.reshape(n_objects, n_objects - 1).mean(axis=1)
            del result
        del pairs
        if self.add_zeroth_order_model:
            scores_zero = self.zero_order_model.predict(X.reshape(-1, n_features))
            scores_zero = scores_zero.reshape(n_instances, n_objects)
            scores = scores + scores_zero
        return scores

    def construct_model(self):
        """
            Construct the :math:`1`-st order and :math:`0`-th order models, which are used to approximate the
            :math:`U_1(x, C(x))` and the :math:`U_0(x)` utilities respectively. For each pair of objects in
            :math:`x_i, x_j \\in Q` :math:`U_1(x, C(x))` we construct :class:`CmpNetCore` with weight sharing to
            approximate a pairwise-matrix. A pairwise matrix with index (i,j) corresponds to the :math:`U_1(x_i,x_j)`
            is a measure of how favorable it is to choose :math:`x_i` over :math:`x_j`. Using this matrix we calculate
            the borda score for each object to calculate :math:`U_1(x, C(x))`. For `0`-th order model we construct
            :math:`\\lvert Q \\lvert` sequential networks whose weights are shared to evaluate the :math:`U_0(x)` for
            each object in the query set :math:`Q`. The output mode is using linear activation.

            Returns
            -------
            model: keras :class:`Model`
                Neural network to learn the FETA utility score
        """

        def create_input_lambda(i):
            return Lambda(lambda x: x[:, i])

        if self.add_zeroth_order_model:
            logger.debug("Create 0th order model")
            zeroth_order_outputs = []
            inputs = []
            for i in range(self.n_objects_fit_):
                x = create_input_lambda(i)(self.input_layer)
                inputs.append(x)
                for hidden in self.hidden_layers_zeroth:
                    x = hidden(x)
                zeroth_order_outputs.append(self.output_node_zeroth(x))
            zeroth_order_scores = concatenate(zeroth_order_outputs)
            logger.debug("0th order model finished")
        logger.debug("Create 1st order model")
        outputs = [list() for _ in range(self.n_objects_fit_)]
        for i, j in combinations(range(self.n_objects_fit_), 2):
            if self.add_zeroth_order_model:
                x1 = inputs[i]
                x2 = inputs[j]
            else:
                x1 = create_input_lambda(i)(self.input_layer)
                x2 = create_input_lambda(j)(self.input_layer)
            x1x2 = concatenate([x1, x2])
            x2x1 = concatenate([x2, x1])

            for hidden in self.hidden_layers:
                x1x2 = hidden(x1x2)
                x2x1 = hidden(x2x1)

            merged_left = concatenate([x1x2, x2x1])
            merged_right = concatenate([x2x1, x1x2])

            n_g = self.output_node(merged_left)
            n_l = self.output_node(merged_right)

            outputs[i].append(n_g)
            outputs[j].append(n_l)
        # convert rows of pairwise matrix to keras layers:
        outputs = [concatenate(x) for x in outputs]

        # compute utility scores:
        scores = [
            Lambda(lambda s: K.mean(s, axis=1, keepdims=True))(x) for x in outputs
        ]
        scores = concatenate(scores)
        logger.debug("1st order model finished")
        if self.add_zeroth_order_model:
            scores = add([scores, zeroth_order_scores])
        model = Model(inputs=self.input_layer, outputs=scores)
        logger.debug("Compiling complete model...")
        model.compile(
            loss=self.loss_function,
            optimizer=self.optimizer_,
            metrics=list(self.metrics),
        )
        return model

    def _pre_fit(self):
        super()._pre_fit()
        self._initialize_optimizer()
        self._initialize_regularizer()
        self.random_state_ = check_random_state(self.random_state)

    def fit(
        self, X, Y, epochs=10, callbacks=None, validation_split=0.1, verbose=0, **kwd
    ):
        """
            Fit a generic preference learning model on a provided set of queries.
            The provided queries can be of a fixed size (numpy arrays).

            Parameters
            ----------
            X : numpy array
                (n_instances, n_objects, n_features)
                Feature vectors of the objects
            Y : numpy array
                (n_instances, n_objects)
                Preferences in form of rankings or choices for given objects
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
        self._construct_layers()

        logger.debug("Enter fit function...")

        X, Y = self.sub_sampling(X, Y)
        self.model_ = self.construct_model()
        logger.debug("Starting gradient descent...")

        self.model_.fit(
            x=X,
            y=Y,
            batch_size=self.batch_size,
            epochs=epochs,
            callbacks=callbacks,
            validation_split=validation_split,
            verbose=verbose,
            **kwd,
        )
        return self

    def sub_sampling(self, X, Y):
        if self.n_objects_fit_ > self.max_number_of_objects:
            bucket_size = int(self.n_objects_fit_ / self.max_number_of_objects)
            idx = self.random_state_.randint(
                bucket_size, size=(len(X), self.n_objects_fit_)
            )
            # TODO: subsampling multiple rankings
            idx += np.arange(start=0, stop=self.n_objects_fit_, step=bucket_size)[
                : self.n_objects_fit_
            ]
            X = X[np.arange(X.shape[0])[:, None], idx]
            Y = Y[np.arange(X.shape[0])[:, None], idx]
            tmp_sort = Y.argsort(axis=-1)
            Y = np.empty_like(Y)
            Y[np.arange(len(X))[:, None], tmp_sort] = np.arange(self.n_objects_fit_)
        return X, Y

    def _predict_scores_fixed(self, X, **kwargs):
        n_objects = X.shape[-2]
        logger.info("For Test instances {} objects {} features {}".format(*X.shape))
        if self.n_objects_fit_ != n_objects:
            scores = self._predict_scores_using_pairs(X, **kwargs)
        else:
            scores = self.model_.predict(X, **kwargs)
        logger.info("Done predicting scores")
        return scores
