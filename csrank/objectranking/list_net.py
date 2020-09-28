import logging

from keras import Input
from keras.layers import concatenate
from keras.layers import Dense
from keras.models import Model
from keras.optimizers import SGD
from keras.regularizers import l2
from sklearn.utils import check_random_state

from csrank.layers import create_input_lambda
from csrank.layers import NormalizedDense
from csrank.learner import Learner
from csrank.losses import plackett_luce_loss
from csrank.metrics import zero_one_rank_loss_for_scores_ties
from csrank.objectranking.object_ranker import ObjectRanker

__all__ = ["ListNet"]
logger = logging.getLogger(__name__)


class ListNet(ObjectRanker, Learner):
    def __init__(
        self,
        n_top=1,
        n_hidden=2,
        n_units=8,
        loss_function=plackett_luce_loss,
        batch_normalization=False,
        kernel_regularizer=l2,
        activation="selu",
        kernel_initializer="lecun_normal",
        optimizer=SGD,
        metrics=(zero_one_rank_loss_for_scores_ties,),
        batch_size=256,
        random_state=None,
        **kwargs,
    ):
        """ Create an instance of the ListNet architecture. ListNet trains a latent utility model based on
            top-k-subrankings of the objects. This network learns a latent utility score for each object in the given
            query set :math:`Q = \\{x_1, \\ldots ,x_n\\}` using the equation :math:`U(x) = F(x, w)` where :math:`w` is the
            weight vector. A listwise loss function like the negative Plackett-Luce likelihood is used for training.
            The ranking for the given query set :math:`Q` is defined as:

            .. math::

                ρ(Q)  = \\operatorname{argsort}_{x \\in Q}  \\; U(x)

            Parameters
            ----------
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
            kernel_regularizer : uninitialized keras regularizer
                Regularizer function applied to all the hidden weight matrices.
            kernel_regularizer__{kwarg}
                Arguments to be passed to the kernel regularizer on initialization, such as kernel_regularizer__l.
            activation : function or string
                Type of activation function to use in each hidden layer
            kernel_initializer : function or string
                Initialization function for the weights of each hidden layer
            optimizer: Class
                Uninitialized optimizer class following the keras optimizer interface.
            optimizer__{kwarg}
                Arguments to be passed to the optimizer on initialization, such as optimizer__lr.
            metrics : list
                List of metrics to evaluate during training (can be
                non-differentiable)
            batch_size : int
                Batch size to use during training
            random_state : int, RandomState instance or None
                Seed of the pseudorandom generator or a RandomState instance
            hidden_dense_layer__{kwarg}
                Arguments to be passed to the Dense layers (or NormalizedDense
                if batch_normalization is enabled). See the keras documentation
                for those classes for available options.
            hidden_dense_layer__{kwarg}
                Arguments to be passed to the Dense layers (or NormalizedDense
                if batch_normalization is enabled). See the keras documentation
                for those classes for available options.

            References
            ----------
                [1] Z. Cao, T. Qin, T. Liu, M. Tsai and H. Li. "Learning to Rank: From Pairwise Approach to Listwise Approach." ICML, 2007.
        """
        self.n_top = n_top
        self.batch_normalization = batch_normalization
        self.activation = activation
        self.metrics = metrics
        self.kernel_regularizer = kernel_regularizer
        self.kernel_initializer = kernel_initializer
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.n_hidden = n_hidden
        self.n_units = n_units

        self.batch_size = batch_size
        self.random_state = random_state
        self._store_kwargs(
            kwargs, {"hidden_dense__", "optimizer__", "kernel_regularizer__"}
        )

    def _construct_layers(self):
        self.input_layer = Input(shape=(self.n_top, self.n_object_features_fit_))
        self.output_node = Dense(
            1, activation="linear", kernel_regularizer=self.kernel_regularizer_
        )
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

    def _create_topk(self, X, Y):
        n_inst, n_obj, n_feat = X.shape
        mask = Y < self.n_top
        X_topk = X[mask].reshape(n_inst, self.n_top, n_feat)
        Y_topk = Y[mask].reshape(n_inst, self.n_top)
        return X_topk, Y_topk

    def _pre_fit(self):
        super()._pre_fit()
        self.random_state_ = check_random_state(self.random_state)
        self._initialize_optimizer()
        self._initialize_regularizer()

    def fit(
        self, X, Y, epochs=10, callbacks=None, validation_split=0.1, verbose=0, **kwd
    ):
        """
            Fit an object ranking learning ListNet on the top-k-subrankings in the provided set of queries. The provided
            queries can be of a fixed size (numpy arrays). For fitting the model we maximize the Plackett-Luce
            likelihood. For example for query set :math:`Q = \\{x_1,x_2,x_3\\}`, the scores are :math:`Q = (s_1,s_2,s_3)`
            and the ranking is :math:`\\pi = (3,1,2)`. The  Plackett-Luce likelihood is defined as:

            .. math::
                P_l(\\pi) = \\frac{s_2}{s_1+s_2+s_3} \\cdot \\frac{s_3}{s_1+s_3} \\cdot \\frac{s_1}{s_1}

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
        self._pre_fit()
        _n_instances, _n_objects, self.n_object_features_fit_ = X.shape
        self._construct_layers()
        logger.debug("Creating top-k dataset")
        X, Y = self._create_topk(X, Y)
        logger.debug("Finished creating the dataset")

        logger.debug("Creating the model")
        self.model_ = self.construct_model()
        logger.debug("Finished creating the model, now fitting...")
        self.model_.fit(
            X,
            Y,
            batch_size=self.batch_size,
            epochs=epochs,
            callbacks=callbacks,
            validation_split=validation_split,
            verbose=verbose,
            **kwd,
        )
        logger.debug("Fitting Complete")
        return self

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
        merged = concatenate(outputs) if len(outputs) > 1 else outputs[0]
        model = Model(inputs=self.input_layer, outputs=merged)
        model.compile(
            loss=self.loss_function,
            optimizer=self.optimizer_,
            metrics=list(self.metrics),
        )
        return model

    @property
    def scoring_model(self):
        """
            Creates a scoring model from the trained ListNet, which predicts the utility scores for given set of objects.
            This network consist of a sequential network which predicts the utility score for each object :math:`x \\in Q`
            using the latent utility function :math:`U(x) = F(x, w)` where :math:`w` is the weights of the model.

            Returns
            -------
            scoring_model: keras model :class:`Model`
                scoring model used to predict utility score for each object
        """
        if not hasattr(self, "scoring_model_"):
            logger.info("Creating scoring model")
            inp = Input(shape=(self.n_object_features_fit_,))
            x = inp
            for hidden_layer in self.hidden_layers:
                x = hidden_layer(x)
            output_score = self.output_node(x)
            self.scoring_model_ = Model(inputs=inp, outputs=output_score)
        return self.scoring_model_

    def _predict_scores_fixed(self, X, **kwargs):
        n_inst, n_obj, n_feat = X.shape
        logger.info("For Test instances {} objects {} features {}".format(*X.shape))
        inp = Input(shape=(n_obj, n_feat))
        lambdas = [create_input_lambda(i)(inp) for i in range(n_obj)]
        scores = concatenate([self.scoring_model(lam) for lam in lambdas])
        model = Model(inputs=inp, outputs=scores)
        return model.predict(X)
