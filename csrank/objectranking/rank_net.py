import logging

import numpy as np
from keras import Input, backend as K, optimizers
from keras.engine import Model
from keras.layers import Dense, Lambda, add
from keras.losses import binary_crossentropy
from keras.metrics import top_k_categorical_accuracy, binary_accuracy
from keras.regularizers import l2
from sklearn.utils import check_random_state

from csrank.layers import NormalizedDense
from csrank.objectranking.constants import THRESHOLD
from csrank.objectranking.object_ranker import ObjectRanker
from csrank.tunable import Tunable
from csrank.util import print_dictionary
from ..dataset_reader.objectranking.util import generate_complete_pairwise_dataset

__all__ = ['RankNet']


class RankNet(ObjectRanker, Tunable):

    def __init__(self, n_object_features, n_hidden=2, n_units=8,
                 loss_function=binary_crossentropy, batch_normalization=True,
                 kernel_regularizer=l2(l=1e-4), non_linearities='relu',
                 optimizer="adam", metrics=[top_k_categorical_accuracy, binary_accuracy],
                 batch_size=256, random_state=None, **kwargs):
        """Create an instance of the RankNet architecture.

        RankNet breaks the rankings into pairwise comparisons and learns a
        latent utility model for the objects.

        Parameters
        ----------
        n_object_features : int
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
        batch_size : int
            Batch size to use during training
        random_state : int, RandomState instance or None
            Seed of the pseudorandom generator or a RandomState instance
        **kwargs
            Keyword arguments for the algorithms

        References
        ----------
        .. [1] Burges, C. et al. (2005, August).
               "Learning to rank using gradient descent.",
               In Proceedings of the 22nd international conference on Machine
               learning (pp. 89-96). ACM.
        .. [2] Burges, C. J. (2010).
               "From ranknet to lambdarank to lambdamart: An overview.",
               Learning, 11(23-581), 81.
        """
        self.logger = logging.getLogger(RankNet.__name__)
        self.n_object_features = n_object_features
        self.batch_normalization = batch_normalization
        self.non_linearities = non_linearities
        self.metrics = metrics
        self.kernel_regularizer = kernel_regularizer
        self.loss_function = loss_function
        self.optimizer = optimizers.get(optimizer)
        self._optimizer_config = self.optimizer.get_config()
        self.n_hidden = n_hidden
        self.n_units = n_units
        self._construct_layers()
        self.threshold_instances = THRESHOLD
        self.batch_size = batch_size
        self.random_state = check_random_state(random_state)

    def _construct_layers(self, **kwargs):
        self.x1 = Input(shape=(self.n_object_features,))
        self.x2 = Input(shape=(self.n_object_features,))
        self.output_node = Dense(1, activation='sigmoid', kernel_regularizer=self.kernel_regularizer)
        self.output_layer_score = Dense(1, activation='linear')
        if self.batch_normalization:
            self.hidden_layers = [NormalizedDense(self.n_units, name="hidden_{}".format(x),
                                                  kernel_regularizer=self.kernel_regularizer,
                                                  activation=self.non_linearities) for x in range(self.n_hidden)]
        else:
            self.hidden_layers = [Dense(self.n_units, name="hidden_{}".format(x),
                                        kernel_regularizer=self.kernel_regularizer,
                                        activation=self.non_linearities)
                                  for x in range(self.n_hidden)]
        assert len(self.hidden_layers) == self.n_hidden

    def fit(self, X, Y, epochs=10, callbacks=None, validation_split=0.1, verbose=0, **kwd):

        X1, X2, Y_single = self.convert_instances(X, Y)

        self.logger.debug('Creating the model')

        output = self.construct_model()

        # Model with input as two objects and output as probability of x1>x2
        self.model = Model(inputs=[self.x1, self.x2], outputs=output)

        self.model.compile(loss=self.loss_function, optimizer=self.optimizer, metrics=self.metrics)
        self.logger.debug('Finished Creating the model, now fitting started')

        self.model.fit([X1, X2], Y_single, batch_size=self.batch_size, epochs=epochs,
                       callbacks=callbacks, validation_split=validation_split,
                       verbose=verbose, **kwd)
        self.scoring_model = self._create_scoring_model()

        self.logger.debug('Fitting Complete')

    def convert_instances(self, X, Y):
        self.logger.debug('Creating the Dataset')
        garbage, X1, X2, garbage, Y_single = generate_complete_pairwise_dataset(X, Y)
        del garbage
        if X1.shape[0] > self.threshold_instances:
            indices = self.random_state.choice(X1.shape[0], self.threshold_instances, replace=False)
            X1 = X1[indices, :]
            X2 = X2[indices, :]
            Y_single = Y_single[indices]
        self.logger.debug('Finished the Dataset')
        return X1, X2, Y_single

    def construct_model(self):
        # weight sharing using same hidden layer for two objects
        enc_x1 = self.hidden_layers[0](self.x1)
        enc_x2 = self.hidden_layers[0](self.x2)
        neg_x2 = Lambda(lambda x: -x)(enc_x2)
        for hidden_layer in self.hidden_layers[1:]:
            enc_x1 = hidden_layer(enc_x1)
            neg_x2 = hidden_layer(neg_x2)
        merged_inputs = add([enc_x1, neg_x2])
        output = self.output_node(merged_inputs)
        return output

    def _create_scoring_model(self):
        inp = Input(shape=(self.n_object_features,))
        x = inp
        for hidden_layer in self.hidden_layers:
            x = hidden_layer(x)
        output_score = self.output_node(x)
        model = Model(inputs=[inp], outputs=output_score)
        return model

    def predict(self, X, **kwargs):
        return super().predict(X, **kwargs)

    def _predict_scores_fixed(self, X, **kwargs):
        # assert X1.shape[1] == self.n_features
        n_instances, n_objects, n_features = X.shape
        self.logger.info("For Test instances {} objects {} features {}".format(n_instances, n_objects, n_features))
        scores = []
        for X1 in X:
            score = self.scoring_model.predict(X1, **kwargs)
            score = score.flatten()
            scores.append(score)
        scores = np.array(scores)
        self.logger.info("Done predicting scores")
        return scores

    def predict_scores(self, X, **kwargs):
        return super().predict_scores(X, **kwargs)

    def predict_pair(self, X1, X2, **kwargs):
        pairwise = np.empty(2)
        pairwise[0] = self.model.predict([X1, X2], **kwargs)
        pairwise[1] = self.model.predict([X2, X1], **kwargs)
        return pairwise

    def evaluate(self, X1_test, X2_test, Y_test, **kwargs):
        return self.model.evaluate([X1_test, X2_test], Y_test, **kwargs)

    def set_tunable_parameters(self, n_hidden=32,
                               n_units=2,
                               reg_strength=1e-4,
                               learning_rate=1e-3,
                               batch_size=128, **point):
        self.n_hidden = n_hidden
        self.n_units = n_units
        self.kernel_regularizer = l2(reg_strength)
        self.batch_size = batch_size
        self.optimizer = self.optimizer.from_config(self._optimizer_config)
        K.set_value(self.optimizer.lr, learning_rate)
        self._construct_layers()
        if len(point) > 0:
            self.logger.warning('This ranking algorithm does not support'
                                ' tunable parameters'
                                ' called: {}'.format(print_dictionary(point)))
