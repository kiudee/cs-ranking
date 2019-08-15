import logging

from keras.optimizers import SGD
from keras.regularizers import l2

from csrank.core.ranknet_core import RankNetCore
from csrank.dataset_reader.objectranking.util import generate_complete_pairwise_dataset
from csrank.objectranking.object_ranker import ObjectRanker

__all__ = ['RankNet']


class RankNet(RankNetCore, ObjectRanker):
    def __init__(self, n_object_features, n_hidden=2, n_units=8, loss_function='binary_crossentropy',
                 batch_normalization=True, kernel_regularizer=l2(l=1e-4), kernel_initializer='lecun_normal',
                 activation='relu', optimizer=SGD(lr=1e-4, nesterov=True, momentum=0.9), metrics=['binary_accuracy'],
                 batch_size=256, random_state=None, **kwargs):
        """ Create an instance of the :class:`RankNetCore` architecture for learning a object ranking function.
            It breaks the preferences into pairwise comparisons and learns a latent utility model for the objects.
            This network learns a latent utility score for each object in the given query set
            :math:`Q = \{x_1, \ldots ,x_n\}` using the equation :math:`U(x) = F(x, w)` where :math:`w` is the weight
            vector. It is estimated using *pairwise preferences* generated from the rankings.
            The ranking for the given query set :math:`Q` is defined as:

            .. math::

                ρ(Q)  = \operatorname{argsort}_{x \in Q}  \; U(x)

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
            kernel_initializer : function or string
                Initialization function for the weights of each hidden layer
            activation : function or string
                Type of activation function to use in each hidden layer
            optimizer : function or string
                Optimizer to use during stochastic gradient descent
            metrics : list
                List of metrics to evaluate during training (can be non-differentiable)
            batch_size : int
                Batch size to use during training
            random_state : int, RandomState instance or None
                Seed of the pseudo-random generator or a RandomState instance
            **kwargs
                Keyword arguments for the algorithms

            References
            ----------
                [1] Burges, C. et al. (2005, August). "Learning to rank using gradient descent.", In Proceedings of the 22nd international conference on Machine learning (pp. 89-96). ACM.
                
                [2] Burges, C. J. (2010). "From ranknet to lambdarank to lambdamart: An overview.", Learning, 11(23-581).

        """
        super().__init__(n_object_features=n_object_features, n_hidden=n_hidden, n_units=n_units,
                         loss_function=loss_function, batch_normalization=batch_normalization,
                         kernel_regularizer=kernel_regularizer, kernel_initializer=kernel_initializer,
                         activation=activation, optimizer=optimizer, metrics=metrics, batch_size=batch_size,
                         random_state=random_state, **kwargs)
        self.logger = logging.getLogger(RankNet.__name__)
        self.logger.info("Initializing network with object features {}".format(self.n_object_features))

    def construct_model(self):
        return super().construct_model()

    def _convert_instances_(self, X, Y):
        self.logger.debug('Creating the Dataset')
        garbage, x1, x2, garbage, y_single = generate_complete_pairwise_dataset(X, Y)
        del garbage
        if x1.shape[0] > self.threshold_instances:
            indices = self.random_state.choice(x1.shape[0], self.threshold_instances, replace=False)
            x1 = x1[indices, :]
            x2 = x2[indices, :]
            y_single = y_single[indices]
        self.logger.debug('Finished the Dataset instances {}'.format(x1.shape[0]))
        return x1, x2, y_single

    def fit(self, X, Y, epochs=10, callbacks=None, validation_split=0.1, verbose=0, **kwd):
        """
            Fit an object ranking learning RankNet model on a provided set of queries. The provided queries can be of a
            fixed size (numpy arrays). For learning this network the binary cross entropy loss function for a pair of
            objects :math:`x_i, x_j \in Q` is defined as:

            .. math::

                C_{ij} =  -\\tilde{P_{ij}}\log(P_{ij}) - (1 - \\tilde{P_{ij}})\log(1 - P{ij}) \enspace,

            where :math:`\\tilde{P_{ij}}` is ground truth probability of the preference of :math:`x_i` over :math:`x_j`.
            :math:`\\tilde{P_{ij}} = 1` if :math:`x_i \succ x_j` else :math:`\\tilde{P_{ij}} = 0`.

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
        super().fit(X, Y, epochs=epochs, callbacks=callbacks, validation_split=validation_split, verbose=verbose, **kwd)

    def _predict_scores_fixed(self, X, **kwargs):
        return super()._predict_scores_fixed(X, **kwargs)

    def predict_scores(self, X, **kwargs):
        return super().predict_scores(X, **kwargs)

    def predict_for_scores(self, scores, **kwargs):
        return ObjectRanker.predict_for_scores(self, scores, **kwargs)

    def predict(self, X, **kwargs):
        return super().predict(X, **kwargs)

    def clear_memory(self, **kwargs):
        super().clear_memory(**kwargs)

    def set_tunable_parameters(self, n_hidden=32, n_units=2, reg_strength=1e-4, learning_rate=1e-3, batch_size=128,
                               **point):
        super().set_tunable_parameters(n_hidden=n_hidden, n_units=n_units, reg_strength=reg_strength,
                                       learning_rate=learning_rate, batch_size=batch_size, **point)
