from itertools import combinations
import math

from keras.losses import binary_crossentropy
import numpy as np
from sklearn.utils import check_random_state
import tensorflow as tf

from csrank.learner import Learner
from csrank.numpy_util import sigmoid
from csrank.util import progress_bar


class FETALinearCore(Learner):
    """Core Learner implementing the First Evaluate then Aggregate approach.

    This implements a linear variant of the FETA approach introduced in
    [PfGuH18]. The idea is to first evaluate each object in each sub-context of
    fixed size with a linear function approximator and then to aggregate these
    evaluations.

    References
    ----------

    .. [PfGuH18] Pfannschmidt, K., Gupta, P., & Hüllermeier, E. (2018). Deep
       architectures for learning context-dependent ranking functions. arXiv
       preprint arXiv:1803.05796. https://arxiv.org/pdf/1803.05796.pdf
    """

    def __init__(
        self,
        learning_rate=1e-3,
        batch_size=256,
        loss_function=binary_crossentropy,
        epochs_drop=50,
        drop=0.01,
        random_state=None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        learning_rate : float
            The learning rate used by the gradient descent optimizer.
        batch_size : int
            The size of the mini-batches used to train the Neural Network.
        loss_function
            The loss function to minimize when training the Neural Network. See
            the functions offered in the keras.losses module for more details.
        epochs_drop: int
            The amount of training epochs after which the learning rate is
            decreased by a factor of `drop`.
        drop: float
            The factor by which to decrease the learning rate every
            `epochs_drop` epochs.
        random_state: np.RandomState
            The random state to use in this object.
        """
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.random_state = random_state
        self.loss_function = loss_function
        self.epochs_drop = epochs_drop
        self.drop = drop
        self.current_lr = None
        self.weight1 = None
        self.bias1 = None
        self.weight2 = None
        self.bias2 = None
        self.w_out = None
        self.optimizer = None
        self.W_last = None

    def _construct_model_(self, n_objects):
        self.X = tf.placeholder(
            "float32", [None, n_objects, self.n_object_features_fit_]
        )
        self.Y = tf.placeholder("float32", [None, n_objects])
        std = 1 / np.sqrt(self.n_object_features_fit_)
        self.b1 = tf.Variable(
            self.random_state_.normal(loc=0, scale=std, size=1), dtype=tf.float32
        )
        self.W1 = tf.Variable(
            self.random_state_.normal(
                loc=0, scale=std, size=2 * self.n_object_features_fit_
            ),
            dtype=tf.float32,
        )
        self.W2 = tf.Variable(
            self.random_state_.normal(
                loc=0, scale=std, size=self.n_object_features_fit_
            ),
            dtype=tf.float32,
        )
        self.b2 = tf.Variable(
            self.random_state_.normal(loc=0, scale=std, size=1), dtype=tf.float32
        )
        self.W_out = tf.Variable(
            self.random_state_.normal(loc=0, scale=std, size=2),
            dtype=tf.float32,
            name="W_out",
        )

        outputs = [list() for _ in range(n_objects)]
        for i, j in combinations(range(n_objects), 2):
            x1 = self.X[:, i]
            x2 = self.X[:, j]
            x1x2 = tf.concat((x1, x2), axis=1)
            x2x1 = tf.concat((x2, x1), axis=1)
            n_g = tf.tensordot(x1x2, self.W1, axes=1) + self.b1
            n_l = tf.tensordot(x2x1, self.W1, axes=1) + self.b1
            outputs[i].append(n_g[:, None])
            outputs[j].append(n_l[:, None])
        outputs = [tf.concat(x, axis=1) for x in outputs]
        outputs = tf.reduce_mean(outputs, axis=-1)
        outputs = tf.transpose(outputs)
        zero_outputs = tf.tensordot(self.X, self.W2, axes=1) + self.b2
        scores = tf.sigmoid(self.W_out[0] * zero_outputs + self.W_out[1] * outputs)
        scores = tf.cast(scores, tf.float32)
        self.loss = self.loss_function(self.Y, scores)
        self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(
            self.loss
        )

    def step_decay(self, epoch):
        """Update the current learning rate.

        Computes the current learning rate based on the initial learning rate,
        the current epoch and the decay speed set by the `epochs_drop` and
        `drop` hyperparameters.

        Parameters
        ----------

        epoch: int
            The current epoch.
        """
        step = math.floor((1 + epoch) / self.epochs_drop)
        self.current_lr = self.learning_rate * math.pow(self.drop, step)
        self.optimizer = tf.train.GradientDescentOptimizer(self.current_lr).minimize(
            self.loss
        )

    def fit(
        self, X, Y, epochs=10, callbacks=None, validation_split=0.1, verbose=0, **kwd
    ):
        """
        Fit the preference learning algorithm on the provided set of queries X
        and preferences Y of those objects. The provided queries and
        corresponding preferences are of a fixed size (numpy arrays).

        Parameters
        ----------
        X : array-like, shape (n_samples, n_objects, n_features)
            Feature vectors of the objects
        Y : array-like, shape (n_samples, n_objects)
            Preferences of the objects in form of rankings or choices
        epochs: int
            The amount of epochs to train for. The training loop will try to
            predict the target variables and adjust its parameters by gradient
            descent `epochs` times.
        """
        self.random_state_ = check_random_state(self.random_state)
        # Global Variables Initializer
        n_instances, self.n_objects_fit_, self.n_object_features_fit_ = X.shape
        self._construct_model_(self.n_objects_fit_)
        init = tf.global_variables_initializer()

        with tf.Session() as tf_session:
            tf_session.run(init)
            self._fit_(X, Y, epochs, n_instances, tf_session, verbose)
            training_cost = tf_session.run(self.loss, feed_dict={self.X: X, self.Y: Y})
            self.logger.info(
                "Fitting completed {} epochs done with loss {}".format(
                    epochs, training_cost.mean()
                )
            )
            self.weight1 = tf_session.run(self.W1)
            self.bias1 = tf_session.run(self.b1)
            self.weight2 = tf_session.run(self.W2)
            self.bias2 = tf_session.run(self.b2)
            self.W_last = tf_session.run(self.W_out)

    def _fit_(self, X, Y, epochs, n_instances, tf_session, verbose):
        try:
            for epoch in range(epochs):
                for start in range(0, n_instances, self.batch_size):
                    end = np.min([start + self.batch_size, n_instances])
                    tf_session.run(
                        self.optimizer,
                        feed_dict={self.X: X[start:end], self.Y: Y[start:end]},
                    )
                    if verbose == 1:
                        progress_bar(end, n_instances, status="Fitting")
                if verbose == 1:
                    c = tf_session.run(self.loss, feed_dict={self.X: X, self.Y: Y})
                    print("Epoch {}: cost {} ".format((epoch + 1), np.mean(c)))
                if (epoch + 1) % 100 == 0:
                    c = tf_session.run(self.loss, feed_dict={self.X: X, self.Y: Y})
                    self.logger.info(
                        "Epoch {}: cost {} ".format((epoch + 1), np.mean(c))
                    )
                self.step_decay(epoch)
        except KeyboardInterrupt:
            self.logger.info("Interrupted")
            c = tf_session.run(self.loss, feed_dict={self.X: X, self.Y: Y})
            self.logger.info("Epoch {}: cost {} ".format((epoch + 1), np.mean(c)))

    def _predict_scores_fixed(self, X, **kwargs):
        """Predict the scores for a given collection of sets of objects of same size.

           Parameters
           ----------
           X : array-like, shape (n_samples, n_objects, n_features)


           Returns
           -------
           Y : array-like, shape (n_samples, n_objects)
               Returns the scores of each of the objects for each of the samples.
        """
        n_instances, n_objects, n_features = X.shape
        assert n_features == self.n_object_features_fit_
        outputs = [list() for _ in range(n_objects)]
        for i, j in combinations(range(n_objects), 2):
            x1 = X[:, i]
            x2 = X[:, j]
            x1x2 = np.concatenate((x1, x2), axis=1)
            x2x1 = np.concatenate((x2, x1), axis=1)
            n_g = np.dot(x1x2, self.weight1) + self.bias1
            n_l = np.dot(x2x1, self.weight1) + self.bias1
            outputs[i].append(n_g)
            outputs[j].append(n_l)
        outputs = np.array(outputs)
        outputs = np.mean(outputs, axis=1).T
        scores_zero = np.dot(X, self.weight2) + self.bias2
        scores = sigmoid(self.W_last[0] * scores_zero + self.W_last[1] * outputs)
        return scores
