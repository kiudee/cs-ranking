import logging

from keras import Input, backend as K, optimizers
from keras.engine import Model
from keras.layers import Dense, concatenate
from keras.regularizers import l2
from sklearn.utils import check_random_state

from csrank.layers import NormalizedDense
from csrank.losses import plackett_luce_loss
from csrank.objectranking.constants import THRESHOLD
from csrank.objectranking.object_ranker import ObjectRanker
from csrank.tunable import Tunable
from csrank.util import print_dictionary, create_input_lambda

__all__ = ["ListNet"]


class ListNet(ObjectRanker, Tunable):



    def __init__(
        self,
        n_features,
        n_top,
        n_hidden=2,
        n_units=8,
        loss_function=plackett_luce_loss,
        batch_normalization=True,
        kernel_regularizer=l2(l=1e-4),
        non_linearities="relu",
        optimizer="adam",
        metrics=None,  # [top_k_categorical_accuracy, binary_accuracy],
        batch_size=256,
        random_state=None,
        **kwargs
    ):
        """ Create an instance of the ListNet architecture.

            ListNet trains a latent utility model based on top-k-subrankings
            of the objects. A listwise loss function like the negative
            Plackett-Luce likelihood is used for training.

            Note: For k=2 we obtain RankNet as a special case.

            Parameters
            ----------
            n_features : int
                Number of features of the object space
            n_top : int
                Size of the top-k-subrankings to consider for training
            n_hidden : int
                Number of hidden layers used in the scoring network
            n_units : int
                Number of hidden units in each layer of the scoring network
            loss_function : function or string
                Listwise loss function which is applied on the top-k objects
            batch_normalization : bool
                Whether to use batch normalization in each hidden layer
            kernel_regularizer : function
                Regularizer function applied to all the hidden weight matrices.
            non_linearities : function or string
                Type of activation function to use in each hidden layer
            optimizer : function or string
                Optimizer to use during stochastic gradient descent
            metrics : list
                List of metrics to evaluate during training (can be
                non-differentiable)
            batch_size : int
                Batch size to use during training
            random_state : int, RandomState instance or None
                Seed of the pseudorandom generator or a RandomState instance
            **kwargs
                Keyword arguments for the algorithms
            References
            ----------
        """
        self.logger = logging.getLogger(ListNet.__name__)
        self.n_features = n_features
        self.n_top = self.n_objects = n_top
        self.batch_normalization = batch_normalization
        self.non_linearities = non_linearities
        self.metrics = metrics
        self.kernel_regularizer = kernel_regularizer
        self.loss_function = loss_function
        self.optimizer = optimizers.get(optimizer)
        self.n_hidden = n_hidden
        self.n_units = n_units
        self._construct_layers(**kwargs)
        self.threshold_instances = THRESHOLD
        self.batch_size = batch_size
        self.random_state = check_random_state(random_state)
        self.model = None
        self._scoring_model = None

    def _construct_layers(self, **kwargs):
        self.input_layer = Input(shape=(self.n_top, self.n_features))
        self.output_node = Dense(
            1, activation="linear", kernel_regularizer=self.kernel_regularizer
        )
        if self.batch_normalization:
            self.hidden_layers = [
                NormalizedDense(
                    self.n_units,
                    name="hidden_{}".format(x),
                    kernel_regularizer=self.kernel_regularizer,
                    activation=self.non_linearities,
                    **kwargs
                )
                for x in range(self.n_hidden)
            ]
        else:
            self.hidden_layers = [
                Dense(
                    self.n_units,
                    name="hidden_{}".format(x),
                    kernel_regularizer=self.kernel_regularizer,
                    activation=self.non_linearities,
                    **kwargs
                )
                for x in range(self.n_hidden)
            ]
        assert len(self.hidden_layers) == self.n_hidden

    def _create_topk(self, X, Y):
        n_inst, n_obj, n_feat = X.shape
        mask = Y < self.n_top
        X_topk = X[mask].reshape(n_inst, self.n_top, n_feat)
        Y_topk = Y[mask].reshape(n_inst, self.n_top)
        return X_topk, Y_topk

    def fit(
        self, X, Y, epochs=10, callbacks=None, validation_split=0.1, verbose=0, **kwd
    ):
        self.n_objects = X.shape[1]
        self.logger.debug("Creating top-k dataset")
        X, Y = self._create_topk(X, Y)
        self.logger.debug("Finished creating the dataset")

        self.logger.debug("Creating the model")
        output = self.construct_model()
        self.model = Model(inputs=self.input_layer, outputs=output)
        self.model.compile(
            loss=self.loss_function, optimizer=self.optimizer, metrics=self.metrics
        )
        self.logger.debug("Finished creating the model, now fitting...")

        self.model.fit(
            X,
            Y,
            batch_size=self.batch_size,
            epochs=epochs,
            callbacks=callbacks,
            validation_split=validation_split,
            verbose=verbose,
            **kwd
        )
        self.logger.debug("Fitting Complete")

    def construct_model(self):
        """ Construct the ListNet architecture.

        Weight sharing guarantees that we have a latent utility model for any
        given object.
        """
        hid = [create_input_lambda(i)(self.input_layer) for i in range(self.n_top)]
        for hidden_layer in self.hidden_layers:
            hid = [hidden_layer(x) for x in hid]
        outputs = [self.output_node(x) for x in hid]
        merged = concatenate(outputs)
        return merged

    @property
    def scoring_model(self):
        if self._scoring_model is None:
            inp = Input(shape=(self.n_features,))
            x = inp
            for hidden_layer in self.hidden_layers:
                x = hidden_layer(x)
            output_score = self.output_node(x)
            self._scoring_model = Model(inputs=inp, outputs=output_score)
        return self._scoring_model

    def predict(self, X, **kwargs):
        return super().predict(X, **kwargs)

    def _predict_scores_fixed(self, X, **kwargs):
        n_inst, n_obj, n_feat = X.shape
        inp = Input(shape=(n_obj, n_feat))
        lambdas = [create_input_lambda(i)(inp) for i in range(n_obj)]
        scores = concatenate([self.scoring_model(lam) for lam in lambdas])
        model = Model(inputs=inp, outputs=scores)
        return model.predict(X)

    def predict_scores(self, X, **kwargs):
        return super().predict_scores(X, **kwargs)

    def set_tunable_parameters(
        self,
        n_hidden=32,
        n_units=2,
        reg_strength=1e-4,
        learning_rate=1e-3,
        batch_size=128,
        **point
    ):
        self.n_hidden = n_hidden
        self.n_units = n_units
        self.kernel_regularizer = l2(reg_strength)
        self.batch_size = batch_size
        K.set_value(self.optimizer.lr, learning_rate)
        self._construct_layers()
        if len(point) > 0:
            self.logger.warning(
                "This ranking algorithm does not support"
                " tunable parameters"
                " called: {}".format(print_dictionary(point))
            )
