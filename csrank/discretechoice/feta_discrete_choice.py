import logging
from itertools import combinations

from keras import Input, Model
from keras import backend as K
from keras.layers import Dense, Lambda, concatenate, Activation
from keras.optimizers import SGD
from keras.regularizers import l2

from csrank.core.feta_network import FETANetwork
from csrank.layers import NormalizedDense
from .discrete_choice import DiscreteObjectChooser


class FETADiscreteChoiceFunction(FETANetwork, DiscreteObjectChooser):
    def __init__(self, n_objects, n_object_features, n_hidden=2, n_units=8, add_zeroth_order_model=False,
                 max_number_of_objects=10, num_subsample=5, loss_function='categorical_hinge',
                 batch_normalization=False, kernel_regularizer=l2(l=1e-4), kernel_initializer='lecun_normal',
                 activation='selu', optimizer=SGD(lr=1e-4, nesterov=True, momentum=0.9),
                 metrics=['categorical_accuracy'], batch_size=256, random_state=None, **kwargs):
        """
            Create a FETA-network architecture for learning the discrete choice functions.
            The first-evaluate-then-aggregate approach approximates the context-dependent utility function using the
            first-order utility function :math:`U_1 \colon \mathcal{X} \\times \mathcal{X} \\rightarrow [0,1]`
            and zeroth-order utility function  :math:`U_0 \colon \mathcal{X} \\rightarrow [0,1]`.
            The scores each object :math:`x` using a context-dependent utility function :math:`U (x, C_i)`:

            .. math::
                 U(x_i, C_i) = U_0(x_i) + \\frac{1}{n-1} \sum_{x_j \in Q \\setminus \{x_i\}} U_1(x_i , x_j) \, .

            Training and prediction complexity is quadratic in the number of objects.
            The discrete choice for the given query set :math:`Q` is defined as:

            .. math::

                dc(Q) := \operatorname{argmax}_{x_i \in Q}  \;  U (x_i, C_i)

            Parameters
            ----------
            n_objects : int
                Number of objects in each query set
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
        self.logger = logging.getLogger(FETADiscreteChoiceFunction.__name__)

    def _construct_layers(self, **kwargs):
        self.input_layer = Input(shape=(self.n_objects, self.n_object_features))
        # Todo: Variable sized input
        # X = Input(shape=(None, n_features))
        if self.batch_normalization:
            if self._use_zeroth_model:
                self.hidden_layers_zeroth = [NormalizedDense(self.n_units, name="hidden_zeroth_{}".format(x), *kwargs)
                                             for x in range(self.n_hidden)]
            self.hidden_layers = [NormalizedDense(self.n_units, name="hidden_{}".format(x), **kwargs)
                                  for x in range(self.n_hidden)]
        else:
            if self._use_zeroth_model:
                self.hidden_layers_zeroth = [Dense(self.n_units, name="hidden_zeroth_{}".format(x), **kwargs)
                                             for x in range(self.n_hidden)]
            self.hidden_layers = [Dense(self.n_units, name="hidden_{}".format(x), **kwargs)
                                  for x in range(self.n_hidden)]
        assert len(self.hidden_layers) == self.n_hidden
        self.output_node = Dense(1, activation="linear", kernel_regularizer=self.kernel_regularizer, name="score")
        if self._use_zeroth_model:
            self.output_node_zeroth = Dense(1, activation="linear", kernel_regularizer=self.kernel_regularizer,
                                            name="zero_score")

    def construct_model(self):
        """
            Construct the :math:`1`-st order and :math:`0`-th order models, which are used to approximate the
            :math:`U_1(x, C(x))` and the :math:`U_0(x)` utilities respectively. For each pair of objects in
            :math:`x_i, x_j \in Q` :math:`U_1(x, C(x))` we construct :class:`CmpNetCore` with weight sharing to
            approximate a pairwise-matrix. A pairwise matrix with index (i,j) corresponds to the :math:`U_1(x_i,x_j)`
            is a measure of how favorable it is to choose :math:`x_i` over :math:`x_j`. Using this matrix we calculate
            the borda score for each object to calculate :math:`U_1(x, C(x))`. For `0`-th order model we construct
            :math:`\lvert Q \lvert` sequential networks whose weights are shared to evaluate the :math:`U_0(x)` for
            each object in the query set :math:`Q`. The output mode is using sigmoid activation.

            Returns
            -------
            model: keras :class:`Model`
                Neural network to learn the FETA utility score
        """

        def create_input_lambda(i):
            return Lambda(lambda x: x[:, i])

        if self._use_zeroth_model:
            self.logger.debug('Create 0th order model')
            zeroth_order_outputs = []
            inputs = []
            for i in range(self.n_objects):
                x = create_input_lambda(i)(self.input_layer)
                inputs.append(x)
                for hidden in self.hidden_layers_zeroth:
                    x = hidden(x)
                zeroth_order_outputs.append(self.output_node_zeroth(x))
            zeroth_order_scores = concatenate(zeroth_order_outputs)
            self.logger.debug('0th order model finished')
        self.logger.debug('Create 1st order model')
        outputs = [list() for _ in range(self.n_objects)]
        for i, j in combinations(range(self.n_objects), 2):
            if self._use_zeroth_model:
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

            N_g = self.output_node(merged_left)
            N_l = self.output_node(merged_right)

            outputs[i].append(N_g)
            outputs[j].append(N_l)
        # convert rows of pairwise matrix to keras layers:
        outputs = [concatenate(x) for x in outputs]
        # compute utility scores:
        sum_fun = lambda s: K.mean(s, axis=1, keepdims=True)
        scores = [Lambda(sum_fun)(x) for x in outputs]
        scores = concatenate(scores)
        self.logger.debug('1st order model finished')

        if self._use_zeroth_model:
            def get_score_object(i):
                return Lambda(lambda x: x[:, i, None])

            concat_scores = [concatenate([get_score_object(i)(scores), get_score_object(i)(zeroth_order_scores)]) for i
                             in range(self.n_objects)]
            weighted_sum = Dense(1, activation='sigmoid', kernel_initializer=self.kernel_initializer,
                                 kernel_regularizer=self.kernel_regularizer, name='weighted_sum')
            scores = []
            for i in range(self.n_objects):
                scores.append(weighted_sum(concat_scores[i]))
            scores = concatenate(scores)
        # if self._use_zeroth_model:
        #     scores = add([scores, zeroth_order_scores])
        # if self._use_zeroth_model:
        #     def expand_dims():
        #         return Lambda(lambda x: x[..., None])
        #
        #     def squeeze_dims():
        #         return Lambda(lambda x: x[:, :, 0])
        #
        #     scores = expand_dims()(scores)
        #     zeroth_order_scores = expand_dims()(zeroth_order_scores)
        #     concat_scores = concatenate([scores, zeroth_order_scores], axis=-1)
        #     weighted_sum = Conv1D(name='weighted_sum', filters=1, kernel_size=(1), strides=1, activation='linear',
        #                          kernel_initializer=self.kernel_initializer, input_shape=(self.n_objects, 2),
        #                          kernel_regularizer=self.kernel_regularizer, use_bias=False)
        #     scores = weighted_sum(concat_scores)
        #     scores = squeeze_dims()(scores)
        if not self._use_zeroth_model:
            scores = Activation('sigmoid')(scores)
        model = Model(inputs=self.input_layer, outputs=scores)
        self.logger.debug('Compiling complete model...')
        model.compile(loss=self.loss_function, optimizer=self.optimizer, metrics=self.metrics)
        return model

    def _create_zeroth_order_model(self):
        inp = Input(shape=(self.n_object_features,))

        x = inp
        for hidden in self.hidden_layers_zeroth:
            x = hidden(x)
        zeroth_output = self.output_node_zeroth(x)

        return Model(inputs=[inp], outputs=Activation('sigmoid')(zeroth_output))

    def fit(self, X, Y, **kwd):
        super().fit(X, Y, **kwd)

    def _predict_scores_fixed(self, X, **kwargs):
        return super()._predict_scores_fixed(X, **kwargs)

    def predict_scores(self, X, **kwargs):
        return super().predict_scores(X, **kwargs)

    def predict_for_scores(self, scores, **kwargs):
        return DiscreteObjectChooser.predict_for_scores(self, scores, **kwargs)

    def predict(self, X, **kwargs):
        return super().predict(X, **kwargs)

    def clear_memory(self, **kwargs):
        self.logger.info("Clearing memory")
        super().clear_memory(**kwargs)

    def set_tunable_parameters(self, n_hidden=32, n_units=2, reg_strength=1e-4, learning_rate=1e-3, batch_size=128,
                               **point):
        super().set_tunable_parameters(n_hidden=n_hidden, n_units=n_units, reg_strength=reg_strength,
                                       learning_rate=learning_rate, batch_size=batch_size, **point)
