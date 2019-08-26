import logging

from keras.optimizers import SGD
from keras.regularizers import l2
from sklearn.model_selection import train_test_split

from csrank.choicefunctions.choice_functions import ChoiceFunctions
from csrank.choicefunctions.util import generate_complete_pairwise_dataset
from csrank.core.cmpnet_core import CmpNetCore


class CmpNetChoiceFunction(CmpNetCore, ChoiceFunctions):
    def __init__(self, n_object_features, n_hidden=2, n_units=8, loss_function='binary_crossentropy',
                 batch_normalization=True, kernel_regularizer=l2(l=1e-4), kernel_initializer='lecun_normal',
                 activation='relu', optimizer=SGD(lr=1e-4, nesterov=True, momentum=0.9), metrics=['binary_accuracy'],
                 batch_size=256, random_state=None, **kwargs):
        """
            Create an instance of the :class:`CmpNetCore` architecture for learning a choice function.
            CmpNet breaks the preferences in form of rankings into pairwise comparisons and learns a pairwise model for
            the each pair of object in the underlying set. For prediction list of objects is converted in pair of
            objects and the pairwise predicate is evaluated using them. The outputs of the network for each pair of
            objects :math:`U(x_1,x_2), U(x_2,x_1)` are evaluated.
            :math:`U(x_1,x_2)` is a measure of how favorable it is to choose :math:`x_1` over :math:`x_2`.
            The utility score of object :math:`x_i` in query set :math:`Q = \{ x_1 , \ldots , x_n \}` is evaluated as:

            .. math::

                U(x_i) = \left\{ \\frac{1}{n-1} \sum_{j \in [n] \setminus \{i\}} U_1(x_i , x_j)\\right\}

            The choice set is defined as:

            .. math::

                c(Q) = \{ x_i \in Q \lvert \, U(x_i) > t \}

            Parameters
            ----------
            n_object_features : int
                Number of features of the object space
            n_hidden : int
                Number of hidden layers used in the scoring network
            n_units : int
                Number of hidden units in each layer of the scoring network
            loss_function : function or string
                Loss function to be used for the binary decision task of the pairwise comparisons
            batch_normalization : bool
                Whether to use batch normalization in each hidden layer
            kernel_regularizer : function
                Regularizer function applied to all the hidden weight matrices.
            activation : function or string
                Type of activation function to use in each hidden layer
            optimizer : function or string
                Optimizer to use during stochastic gradient descent
            metrics : list
                List of metrics to evaluate during training (can be non-differentiable)
            batch_size : int
                Batch size to use during training
            random_state : int, RandomState instance or None
                Seed of the pseudorandom generator or a RandomState instance
            **kwargs
                Keyword arguments for the algorithms

            References
            ----------
                [1] Leonardo Rigutini, Tiziano Papini, Marco Maggini, and Franco Scarselli. 2011. SortNet: Learning to Rank by a Neural Preference Function. IEEE Trans. Neural Networks 22, 9 (2011), 1368–1380. https://doi.org/10.1109/TNN.2011.2160875

        """
        super().__init__(n_object_features=n_object_features, n_hidden=n_hidden, n_units=n_units,
                         loss_function=loss_function, batch_normalization=batch_normalization,
                         kernel_regularizer=kernel_regularizer, kernel_initializer=kernel_initializer,
                         activation=activation, optimizer=optimizer, metrics=metrics, batch_size=batch_size,
                         random_state=random_state, **kwargs)
        self.logger = logging.getLogger(CmpNetChoiceFunction.__name__)
        self.logger.info("Initializing network with object features {}".format(self.n_object_features))
        self.threshold = 0.5

    def _convert_instances_(self, X, Y):
        self.logger.debug('Creating the Dataset')
        x1, x2, garbage, y_double, garbage = generate_complete_pairwise_dataset(X, Y)
        del garbage
        if x1.shape[0] > self.threshold_instances:
            indices = self.random_state.choice(x1.shape[0], self.threshold_instances, replace=False)
            x1 = x1[indices, :]
            x2 = x2[indices, :]
            y_double = y_double[indices, :]
        self.logger.debug('Finished the Dataset instances {}'.format(x1.shape[0]))
        return x1, x2, y_double

    def construct_model(self):
        return super().construct_model()

    def fit(self, X, Y, epochs=10, callbacks=None, validation_split=0.1, tune_size=0.1, thin_thresholds=1, verbose=0,
            **kwd):
        """
            Fit a CmptNet model for learning a choice fucntion on the provided set of queries X and preferences Y of
            those objects. The provided queries and corresponding preferences are of a fixed size (numpy arrays). For
            learning this network the binary cross entropy loss function for a pair of objects :math:`x_i, x_j \in Q`
            is defined as:

            .. math::

                C_{ij} =  -\\tilde{P_{ij}}(0)\\cdot \log(U(x_i,x_j)) - \\tilde{P_{ij}}(1) \\cdot \log(U(x_j,x_i)) \ ,

            where :math:`\\tilde{P_{ij}}` is ground truth probability of the preference of :math:`x_i` over :math:`x_j`.
            :math:`\\tilde{P_{ij}} = (1,0)` if :math:`x_i \succ x_j` else :math:`\\tilde{P_{ij}} = (0,1)`.

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
            tune_size: float (range : [0,1])
                Percentage of instances to split off to tune the threshold for the choice function
            thin_thresholds: int
                The number of instances of scores to skip while tuning the threshold
            verbose : bool
                Print verbose information
            **kwd :
                Keyword arguments for the fit function
        """
        if tune_size > 0:
            X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=tune_size, random_state=self.random_state)
            try:
                super().fit(X_train, Y_train, epochs, callbacks,
                            validation_split, verbose, **kwd)
            finally:
                self.logger.info('Fitting utility function finished. Start tuning threshold.')
        else:
            super().fit(X, Y, epochs, callbacks, validation_split, verbose,
                        **kwd)
            self.threshold = 0.5

    def _predict_scores_fixed(self, X, **kwargs):
        return super()._predict_scores_fixed(X, **kwargs)

    def predict_scores(self, X, **kwargs):
        return super().predict_scores(X, **kwargs)

    def predict_for_scores(self, scores, **kwargs):
        return ChoiceFunctions.predict_for_scores(self, scores, **kwargs)

    def predict(self, X, **kwargs):
        return super().predict(X, **kwargs)

    def clear_memory(self, **kwargs):
        super().clear_memory(**kwargs)

    def set_tunable_parameters(self, n_hidden=32, n_units=2, reg_strength=1e-4, learning_rate=1e-3, batch_size=128,
                               **point):
        super().set_tunable_parameters(n_hidden=n_hidden, n_units=n_units, reg_strength=reg_strength,
                                       learning_rate=learning_rate, batch_size=batch_size, **point)
