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
        """ Create an instance of the RankNet architecture.
            RankNet breaks the preferences into pairwise comparisons and learns a latent utility model for the objects.

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
            .. [1] Burges, C. et al. (2005, August)."Learning to rank using gradient descent.",
                   In Proceedings of the 22nd international conference on Machine learning (pp. 89-96). ACM.
            .. [2] Burges, C. J. (2010). "From ranknet to lambdarank to lambdamart: An overview.",
                   Learning, 11(23-581), 81.
        """
        super().__init__(n_object_features=n_object_features, n_hidden=n_hidden, n_units=n_units,
                         loss_function=loss_function, batch_normalization=batch_normalization,
                         kernel_regularizer=kernel_regularizer, kernel_initializer=kernel_initializer,
                         activation=activation, optimizer=optimizer, metrics=metrics, batch_size=batch_size,
                         random_state=random_state, **kwargs)
        self.logger = logging.getLogger(RankNet.__name__)
        self.logger.info("Initializing network with object features {}".format(self.n_object_features))

    def _convert_instances(self, X, Y):
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

    def fit(self, X, Y, **kwd):
        super().fit(X, Y, **kwd)

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
