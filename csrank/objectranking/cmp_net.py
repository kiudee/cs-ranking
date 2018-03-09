import logging
import operator
from collections import OrderedDict
from itertools import combinations, permutations

import numpy as np
from keras import backend as K, optimizers
from keras.layers import Dense, Input, concatenate
from keras.losses import binary_crossentropy
from keras.metrics import top_k_categorical_accuracy, binary_accuracy
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2
from scipy.stats import rankdata
from sklearn.utils import check_random_state

from csrank.callbacks import EarlyStoppingWithWeights
from csrank.constants import REGULARIZATION_FACTOR, LEARNING_RATE, BATCH_SIZE, \
    LR_DEFAULT_RANGE, REGULARIZATION_FACTOR_DEFAULT_RANGE, \
    BATCH_SIZE_DEFAULT_RANGE, EARLY_STOPPING_PATIENCE, EARLY_STOPPING_PATIENCE_DEFAULT_RANGE
from csrank.layers import NormalizedDense
from csrank.objectranking.constants import THRESHOLD, N_HIDDEN_LAYERS, N_HIDDEN_UNITS, \
    N_HIDDEN_LAYERS_DEFAULT_RANGES, \
    N_UNITS_DEFAULT_RANGES
from csrank.objectranking.object_ranker import ObjectRanker
from csrank.tunable import Tunable
from csrank.util import tunable_parameters_ranges
from ..dataset_reader.objectranking.util import generate_complete_pairwise_dataset

__all__ = ['CmpNet']


class CmpNet(ObjectRanker, Tunable):
    _tunable = None
    _use_early_stopping = None

    def __init__(self, n_features, n_hidden=2, n_units=8,
                 loss_function=binary_crossentropy, batch_normalization=True,
                 kernel_regularizer=l2(l=0.02), non_linearities='relu',
                 optimizer=Adam(), metrics=[top_k_categorical_accuracy, binary_accuracy],
                 use_early_stopping=False, es_patience=300, batch_size=256, random_state=None, **kwargs):
        """
            Create an instance of the CmpNet architecture.

            CmpNet breaks the rankings into pairwise comparisons and learns a pairwise model for the each pair of object in the ranking.
            For prediction list of objects is converted in pair of objects and the pairwise predicate is evaluated using them.
            The outputs of the network for each pair of objects $U(x_1,x_2), U(x_2,x_1)$ are evaluated.
            $U(x_1,x_2) is a measure of how favorable it is for $x_1$ than $x_2$.
            Ranking for the given set of objects $Q = \{ \vec{x}_1 , \ldots , \vec{x}_n \}$  is evaluted as follows:
            \begin{equation}
                \rho(\mathcal{Q}) = \argsort_{i \in [n]}  \; \biggl\{\frac{1}{n-1} \sum_{j \in [n] \setminus \{i\}} U_1(\vec{x}_i , \vec{x}_j)
            \end{equation}

            Parameters
            ----------
            n_features : int
                Number of features of the object space
            n_hidden : int
                Number of hidden layers used in the scoring network
            n_units : int
                Number of hidden units in each layer of the scoring network
            loss_function : function or string
                Loss function to be used for the binary decision task of the
                pairwise comparisons
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
            use_early_stopping : bool
                If True, stop the training early, if no progress has been made for
                es_patience many iterations
            es_patience : int
                If early stopping is enabled, wait for this many iterations without
                progress until stopping the training
            batch_size : int
                Batch size to use during training
            random_state : int, RandomState instance or None
                Seed of the pseudorandom generator or a RandomState instance
            **kwargs
                Keyword arguments for the algorithms

            References
            ----------
            .. [1] Leonardo Rigutini, Tiziano Papini, Marco Maggini, and Franco Scarselli. 2011.
               SortNet: Learning to Rank by a Neural Preference Function.
               IEEE Trans. Neural Networks 22, 9 (2011), 1368–1380. https://doi.org/10.1109/TNN.2011.2160875
        """
        self.logger = logging.getLogger("CmpNet")
        self.n_features = n_features
        self.batch_normalization = batch_normalization
        self.non_linearities = non_linearities
        self.early_stopping = EarlyStoppingWithWeights(patience=es_patience)
        self._use_early_stopping = use_early_stopping

        self.batch_size = batch_size

        self.metrics = metrics
        self.kernel_regularizer = kernel_regularizer
        self.loss_function = loss_function

        self.optimizer = optimizers.get(optimizer)
        self._construct_layers(n_hidden, n_units)
        self.threshold_instances = THRESHOLD
        self.random_state = check_random_state(random_state)

    def _construct_layers(self, n_hidden=2, n_units=16, **kwargs):

        self.output_node = Dense(1, activation='sigmoid',
                                 kernel_regularizer=self.kernel_regularizer)
        self.x1 = Input(shape=(self.n_features,))
        self.x2 = Input(shape=(self.n_features,))
        if self.batch_normalization:
            self.hidden_layers = [
                NormalizedDense(n_units, name="hidden_{}".format(x),
                                kernel_regularizer=self.kernel_regularizer,
                                activation=self.non_linearities
                                )
                for x in range(n_hidden)
            ]
        else:
            self.hidden_layers = [
                Dense(n_units, name="hidden_{}".format(x),
                      kernel_regularizer=self.kernel_regularizer,
                      activation=self.non_linearities)
                for x in range(n_hidden)
            ]
        assert len(self.hidden_layers) == n_hidden

    def fit(self, X, Y, epochs=10, log_callbacks=None,
            validation_split=0.1, verbose=0, **kwd):

        self.logger.debug('Creating the Dataset')
        garbage, X1, X2, Y_double, garbage = generate_complete_pairwise_dataset(X, Y)
        del garbage
        self.logger.debug('Finished the Dataset')
        if (X1.shape[0] > self.threshold_instances):
            indicies = self.random_state.choice(X1.shape[0], self.threshold_instances, replace=False)
            X1 = X1[indicies, :]
            X2 = X2[indicies, :]
            Y_double = Y_double[indicies, :]

        merged_output = self.construct_model()

        callbacks = []
        if log_callbacks is None:
            log_callbacks = []
        callbacks.extend(log_callbacks)
        callbacks = self.set_init_lr_callback(callbacks)
        if self._use_early_stopping:
            callbacks.append(self.early_stopping)

        self.logger.info("Callbacks {}".format(', '.join([c.__name__ for c in callbacks])))
        self.model = Model(inputs=[self.x1, self.x2], outputs=merged_output)
        self.model.compile(loss=self.loss_function, optimizer=self.optimizer, metrics=self.metrics)

        self.logger.debug('Finished Creating the model, now fitting started')

        self.model.fit([X1, X2], Y_double, batch_size=self.batch_size, epochs=epochs,
                       callbacks=callbacks, validation_split=validation_split,
                       verbose=verbose, **kwd)
        self.logger.debug('Fitting Complete')

    def construct_model(self):
        x1x2 = concatenate([self.x1, self.x2])
        x2x1 = concatenate([self.x2, self.x1])
        self.logger.debug('Creating the model')
        for hidden in self.hidden_layers:
            x1x2 = hidden(x1x2)
            x2x1 = hidden(x2x1)
        merged_left = concatenate([x1x2, x2x1])
        merged_right = concatenate([x2x1, x1x2])
        N_g = self.output_node(merged_left)
        N_l = self.output_node(merged_right)
        merged_output = concatenate([N_g, N_l])
        return merged_output

    def _predict_scores_fixed(self, X, **kwargs):
        n_instances, n_objects, n_features = X.shape
        self.logger.info("For Test instances {} objects {} features {}".format(n_instances, n_objects, n_features))
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
        self.logger.info("Done predicting scores")

        return scores

    def predict_scores(self, X, **kwargs):
        return super().predict_scores(X, **kwargs)

    def predict(self, X, **kwargs):
        return ObjectRanker.predict(self, X, **kwargs)

    def predict_pair(self, a, b, **kwargs):
        return self.model.predict([a, b], **kwargs)

    def evaluate(self, X1_test, X2_test, Y_test, **kwd):
        return self.model.evaluate([X1_test, X2_test], Y_test, **kwd)

    @classmethod
    def set_tunable_parameter_ranges(cls, param_ranges_dict):
        logger = logging.getLogger("CmpNet")
        return tunable_parameters_ranges(cls, logger, param_ranges_dict)

    def set_tunable_parameters(self, point):
        named = Tunable.set_tunable_parameters(self, point)
        hidden_layers_created = False
        for name, param in named.items():
            if name in [N_HIDDEN_LAYERS, N_HIDDEN_UNITS, REGULARIZATION_FACTOR]:
                self.kernel_regularizer = l2(l=named[REGULARIZATION_FACTOR])
                if not hidden_layers_created:
                    self._construct_layers(**named)
                hidden_layers_created = True
            elif name == LEARNING_RATE:
                K.set_value(self.optimizer.lr, param)
            elif name == EARLY_STOPPING_PATIENCE:
                self.early_stopping.patience = param
            elif name == BATCH_SIZE:
                self.batch_size = param
            else:
                self.logger.warning(
                    'This ranking algorithm does not support a tunable parameter called {}'.format(name))

    @classmethod
    def tunable_parameters(cls):
        if cls._tunable is None:
            cls._tunable = OrderedDict([
                (N_HIDDEN_LAYERS, N_HIDDEN_LAYERS_DEFAULT_RANGES),
                (N_HIDDEN_UNITS, N_UNITS_DEFAULT_RANGES),
                (LEARNING_RATE, LR_DEFAULT_RANGE),
                (REGULARIZATION_FACTOR, REGULARIZATION_FACTOR_DEFAULT_RANGE),
                (BATCH_SIZE, BATCH_SIZE_DEFAULT_RANGE),
            ])
            if cls._use_early_stopping:
                cls._tunable[EARLY_STOPPING_PATIENCE] = EARLY_STOPPING_PATIENCE_DEFAULT_RANGE
