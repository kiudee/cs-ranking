import logging

import tensorflow as tf
from keras import Input, backend as K, optimizers
from keras.layers import Dense, concatenate
from keras.models import Model
from keras.optimizers import SGD
from keras.regularizers import l2
from sklearn.utils import check_random_state

from csrank.constants import allowed_dense_kwargs
from csrank.layers import NormalizedDense, create_input_lambda
from csrank.learner import Learner
from csrank.losses import plackett_luce_loss
from csrank.metrics import zero_one_rank_loss_for_scores_ties
from csrank.objectranking.object_ranker import ObjectRanker
from csrank.util import print_dictionary

__all__ = ["ListNet"]


class ListNet(Learner, ObjectRanker):

    def __init__(self, n_object_features, n_top, n_hidden=2, n_units=8, loss_function=plackett_luce_loss,
                 batch_normalization=False, kernel_regularizer=l2(l=1e-4), activation="selu",
                 kernel_initializer='lecun_normal', optimizer=SGD(lr=1e-4, nesterov=True, momentum=0.9),
                 metrics=[zero_one_rank_loss_for_scores_ties], batch_size=256, random_state=None, **kwargs):
        """ Create an instance of the ListNet architecture. ListNet trains a latent utility model based on
            top-k-subrankings of the objects. This network learns a latent utility score for each object in the given
            query set :math:`Q = \{x_1, \ldots ,x_n\}` using the equation :math:`U(x) = F(x, w)` where :math:`w` is the
            weight vector. A listwise loss function like the negative Plackett-Luce likelihood is used for training.
            The ranking for the given query set :math:`Q` is defined as:

            .. math::

                ρ(Q)  = \operatorname{argsort}_{x \in Q}  \; U(x)

            Parameters
            ----------
            n_object_features : int
                Number of features of the object space
            n_top : int
                Size of the top-k-subrankings to consider for training
            hash_file: str
                File path of the model where the weights are stored to get the predictions after clearing the memory
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
            activation : function or string
                Type of activation function to use in each hidden layer
            kernel_initializer : function or string
                Initialization function for the weights of each hidden layer
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
                [1] Z. Cao, T. Qin, T. Liu, M. Tsai and H. Li. "Learning to Rank: From Pairwise Approach to Listwise Approach." ICML, 2007.
        """
        self.logger = logging.getLogger(ListNet.__name__)
        self.n_object_features = n_object_features
        self.n_objects = n_top
        self.n_top = n_top
        self.batch_normalization = batch_normalization
        self.activation = activation
        self.metrics = metrics
        self.kernel_regularizer = kernel_regularizer
        self.kernel_initializer = kernel_initializer
        self.loss_function = loss_function
        self.optimizer = optimizers.get(optimizer)
        self._optimizer_config = self.optimizer.get_config()
        self.n_hidden = n_hidden
        self.n_units = n_units
        keys = list(kwargs.keys())
        for key in keys:
            if key not in allowed_dense_kwargs:
                del kwargs[key]
        self.kwargs = kwargs
        self._construct_layers(kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.kernel_initializer,
                               activation=self.activation, **self.kwargs)

        self.threshold_instances = int(1e10)
        self.batch_size = batch_size
        self.random_state = check_random_state(random_state)
        self.hash_file = None
        self.model = None
        self._scoring_model = None

    def _construct_layers(self, **kwargs):
        self.input_layer = Input(shape=(self.n_top, self.n_object_features))
        self.output_node = Dense(1, activation="linear", kernel_regularizer=self.kernel_regularizer)
        if self.batch_normalization:
            self.hidden_layers = [
                NormalizedDense(self.n_units, name="hidden_{}".format(x), **kwargs)
                for x in range(self.n_hidden)]
        else:
            self.hidden_layers = [
                Dense(self.n_units, name="hidden_{}".format(x), **kwargs)
                for x in range(self.n_hidden)]
        assert len(self.hidden_layers) == self.n_hidden

    def _create_topk(self, X, Y):
        n_inst, n_obj, n_feat = X.shape
        mask = Y < self.n_top
        X_topk = X[mask].reshape(n_inst, self.n_top, n_feat)
        Y_topk = Y[mask].reshape(n_inst, self.n_top)
        return X_topk, Y_topk

    def fit(self, X, Y, epochs=10, callbacks=None, validation_split=0.1, verbose=0, **kwd):
        """
            Fit an object ranking learning ListNet on the top-k-subrankings in the provided set of queries. The provided
            queries can be of a fixed size (numpy arrays). For fitting the model we maximize the Plackett-Luce
            likelihood. For example for query set :math:`Q = \{x_1,x_2,x_3\}`, the scores are :math:`Q = (s_1,s_2,s_3)`
            and the ranking is :math:`\pi = (3,1,2)`. The  Plackett-Luce likelihood is defined as:

            .. math::
                P_l(\pi) = \\frac{s_2}{s_1+s_2+s_3} \cdot \\frac{s_3}{s_1+s_3} \cdot \\frac{s_1}{s_1}

            Note: For k=2 we obtain :class:`RankNet` as a special case.

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
        self.n_objects = X.shape[1]
        self.logger.debug("Creating top-k dataset")
        X, Y = self._create_topk(X, Y)
        self.logger.debug("Finished creating the dataset")

        self.logger.debug("Creating the model")
        self.model = self.construct_model()
        self.logger.debug("Finished creating the model, now fitting...")
        self.model.fit(X, Y, batch_size=self.batch_size, epochs=epochs, callbacks=callbacks,
                       validation_split=validation_split, verbose=verbose, **kwd)
        self.logger.debug("Fitting Complete")

    def construct_model(self):
        """
            Construct the ListNet architecture which takes topk-subrankings from the given queries and minimize a
            listwise loss on the utility scores of top objects. Weight sharing guarantees that we learn the shared
            weights :math:`w` of the latent utility function :math:`U(x) = F(x, w)`.

            Returns
            -------
            model: keras model :class:`Model`
                ListNet model used to learn the utiliy function using the top-k-subrankings in the provided set of queries.
        """
        hid = [create_input_lambda(i)(self.input_layer) for i in range(self.n_top)]
        for hidden_layer in self.hidden_layers:
            hid = [hidden_layer(x) for x in hid]
        outputs = [self.output_node(x) for x in hid]
        merged = concatenate(outputs)
        model = Model(inputs=self.input_layer, outputs=merged)
        model.compile(loss=self.loss_function, optimizer=self.optimizer, metrics=self.metrics)
        return model

    @property
    def scoring_model(self):
        """
            Creates a scoring model from the trained ListNet, which predicts the utility scores for given set of objects.
            This network consist of a sequential network which predicts the utility score for each object :math:`x \in Q`
            using the latent utility function :math:`U(x) = F(x, w)` where :math:`w` is the weights of the model.

            Returns
            -------
            scoring_model: keras model :class:`Model`
                scoring model used to predict utility score for each object
        """
        if self._scoring_model is None:
            self.logger.info('Creating scoring model')
            inp = Input(shape=(self.n_object_features,))
            x = inp
            for hidden_layer in self.hidden_layers:
                x = hidden_layer(x)
            output_score = self.output_node(x)
            self._scoring_model = Model(inputs=inp, outputs=output_score)
        return self._scoring_model

    def _predict_scores_fixed(self, X, **kwargs):
        n_inst, n_obj, n_feat = X.shape
        self.logger.info("For Test instances {} objects {} features {}".format(*X.shape))
        inp = Input(shape=(n_obj, n_feat))
        lambdas = [create_input_lambda(i)(inp) for i in range(n_obj)]
        scores = concatenate([self.scoring_model(lam) for lam in lambdas])
        model = Model(inputs=inp, outputs=scores)
        return model.predict(X)

    def predict_scores(self, X, **kwargs):
        return super().predict_scores(X, **kwargs)

    def predict_for_scores(self, scores, **kwargs):
        return ObjectRanker.predict_for_scores(self, scores, **kwargs)

    def predict(self, X, **kwargs):
        return super().predict(X, **kwargs)

    def clear_memory(self, **kwargs):
        """
            Clear the memory, restores the currently fitted model back to prevent memory leaks.

            Parameters
            ----------
            **kwargs :
                Keyword arguments for the function
        """
        if self.hash_file is not None:
            self.model.save_weights(self.hash_file)
            K.clear_session()
            sess = tf.Session()
            K.set_session(sess)
            self._scoring_model = None
            self.optimizer = self.optimizer.from_config(self._optimizer_config)
            self._construct_layers(kernel_regularizer=self.kernel_regularizer,
                                   kernel_initializer=self.kernel_initializer,
                                   activation=self.activation, **self.kwargs)
            self.model = self.construct_model()
            self.model.load_weights(self.hash_file)
        else:
            self.logger.info("Cannot clear the memory")

    def set_tunable_parameters(self, n_hidden=32, n_units=2, reg_strength=1e-4, learning_rate=1e-3, batch_size=128,
                               **point):
        """
            Set tunable parameters of the ListNet network to the values provided.

            Parameters
            ----------
            n_hidden: int
                Number of hidden layers used in the scoring network
            n_units: int
                Number of hidden units in each layer of the scoring network
            reg_strength: float
                Regularization strength of the regularizer function applied to the `kernel` weights matrix
            learning_rate: float
                Learning rate of the stochastic gradient descent algorithm used by the network
            batch_size: int
                Batch size to use during training
            point: dict
                Dictionary containing parameter values which are not tuned for the network
        """
        self.n_hidden = n_hidden
        self.n_units = n_units
        self.kernel_regularizer = l2(reg_strength)
        self.batch_size = batch_size
        self.optimizer = self.optimizer.from_config(self._optimizer_config)
        K.set_value(self.optimizer.lr, learning_rate)
        self._construct_layers(kernel_regularizer=self.kernel_regularizer,
                               kernel_initializer=self.kernel_initializer,
                               activation=self.activation, **self.kwargs)

        self._scoring_model = None
        if len(point) > 0:
            self.logger.warning("This ranking algorithm does not support "
                                "tunable parameters called: {}".format(print_dictionary(point)))
