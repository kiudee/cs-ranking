import math

import numpy as np
import tensorflow as tf
from keras.losses import binary_crossentropy
from sklearn.utils import check_random_state

from csrank.learner import Learner
from csrank.numpy_util import sigmoid
from csrank.util import progress_bar, print_dictionary


class FATELinearCore(Learner):
    def __init__(self, n_object_features, n_objects, n_hidden_set_units=32, learning_rate=1e-3, batch_size=256,
                 loss_function=binary_crossentropy, epochs_drop=300, drop=0.1, random_state=None, **kwargs):
        self.n_hidden_set_units = n_hidden_set_units
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.random_state = check_random_state(random_state)
        self.n_object_features = n_object_features
        self.loss_function = loss_function
        self.n_objects = n_objects
        self.epochs_drop = epochs_drop
        self.drop = drop
        self.current_lr = None
        self.weight1 = None
        self.bias1 = None
        self.weight2 = None
        self.bias2 = None
        self.optimizer = None

    def _construct_model_(self, n_objects):
        self.X = tf.placeholder("float32", [None, n_objects, self.n_object_features])
        self.Y = tf.placeholder("float32", [None, n_objects])
        std = 1 / np.sqrt(self.n_object_features)
        self.b1 = tf.Variable(self.random_state.normal(loc=0, scale=std, size=self.n_hidden_set_units),
                              dtype=tf.float32)
        self.W1 = tf.Variable(
            self.random_state.normal(loc=0, scale=std, size=(self.n_object_features, self.n_hidden_set_units)),
            dtype=tf.float32)
        self.W2 = tf.Variable(
            self.random_state.normal(loc=0, scale=std, size=(self.n_object_features + self.n_hidden_set_units)),
            dtype=tf.float32)
        self.b2 = tf.Variable(self.random_state.normal(loc=0, scale=std, size=1), dtype=tf.float32)

        set_rep = tf.reduce_mean(tf.tensordot(self.X, self.W1, axes=1), axis=1) + self.b1

        self.set_rep = tf.reshape(tf.tile(set_rep, tf.constant([1, n_objects])),
                                  (-1, n_objects, self.n_hidden_set_units))
        self.X_con = tf.concat([self.X, self.set_rep], axis=-1)
        scores = tf.sigmoid(tf.tensordot(self.X_con, self.W2, axes=1) + self.b2)
        scores = tf.cast(scores, tf.float32)
        self.loss = self.loss_function(self.Y, scores)
        self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)

    def step_decay(self, epoch):
        step = math.floor((1 + epoch) / self.epochs_drop)
        self.current_lr = self.learning_rate * math.pow(self.drop, step)
        self.optimizer = tf.train.GradientDescentOptimizer(self.current_lr).minimize(self.loss)

    def fit(self, X, Y, epochs=10, callbacks=None, validation_split=0.1, verbose=0, **kwd):
        # Global Variables Initializer
        n_instances, n_objects, n_features = X.shape
        assert n_features == self.n_object_features
        self._construct_model_(n_objects)
        init = tf.global_variables_initializer()

        with tf.Session() as tf_session:
            tf_session.run(init)
            self._fit_(X, Y, epochs, n_instances, tf_session, verbose)
            training_cost = tf_session.run(self.loss, feed_dict={self.X: X, self.Y: Y})
            self.logger.info("Fitting completed {} epochs done with loss {}".format(epochs, training_cost.mean()))
            self.weight1 = tf_session.run(self.W1)
            self.bias1 = tf_session.run(self.b1)
            self.weight2 = tf_session.run(self.W2)
            self.bias2 = tf_session.run(self.b2)

    def _fit_(self, X, Y, epochs, n_instances, tf_session, verbose):
        try:
            for epoch in range(epochs):
                for start in range(0, n_instances, self.batch_size):
                    end = np.min([start + self.batch_size, n_instances])
                    tf_session.run(self.optimizer, feed_dict={self.X: X[start:end], self.Y: Y[start:end]})
                    if verbose == 1:
                        progress_bar(end, n_instances, status='Fitting')
                if verbose == 1:
                    c = tf_session.run(self.loss, feed_dict={self.X: X, self.Y: Y})
                    print("Epoch {}: cost {} ".format((epoch + 1), np.mean(c)))
                if (epoch + 1) % 100 == 0:
                    c = tf_session.run(self.loss, feed_dict={self.X: X, self.Y: Y})
                    self.logger.info("Epoch {}: cost {} ".format((epoch + 1), np.mean(c)))
                self.step_decay(epoch)
        except KeyboardInterrupt:
            self.logger.info("Interrupted")
            c = tf_session.run(self.loss, feed_dict={self.X: X, self.Y: Y})
            self.logger.info("Epoch {}: cost {} ".format((epoch + 1), np.mean(c)))

    def _predict_scores_fixed(self, X, **kwargs):
        n_instances, n_objects, n_features = X.shape
        assert n_features == self.n_object_features
        rep = np.mean(np.dot(X, self.weight1), axis=1) + self.bias1
        rep = np.tile(rep[:, np.newaxis, :], (1, n_objects, 1))
        X_n = np.concatenate((X, rep), axis=2)
        scores = np.dot(X_n, self.weight2) + self.bias2
        scores = sigmoid(scores)
        return scores

    def set_tunable_parameters(self, n_hidden_set_units=32, learning_rate=1e-3, batch_size=128, epochs_drop=300,
                               drop=0.1, **point):
        """
            Set tunable parameters of the FETA-network to the values provided.

            Parameters
            ----------
            n_hidden_set_units: int
                Number of hidden units in each layer of the scoring network
            learning_rate: float
                Learning rate of the stochastic gradient descent algorithm used by the network
            batch_size: int
                Batch size to use during training
            epochs_drop: int
                The epochs after which the learning rate is decreased
            drop: float
                The percentage with which the learning rate is decreased
            point: dict
                Dictionary containing parameter values which are not tuned for the network
        """
        self.n_hidden_set_units = n_hidden_set_units
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self._construct_model_(self.n_objects)
        self.epochs_drop = epochs_drop
        self.drop = drop
        if len(point) > 0:
            self.logger.warning('This ranking algorithm does not support tunable parameters'
                                ' called: {}'.format(print_dictionary(point)))
