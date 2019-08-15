import math
from itertools import combinations

import numpy as np
import tensorflow as tf
from csrank.learner import Learner
from csrank.numpy_util import sigmoid
from csrank.util import progress_bar, print_dictionary
from keras.losses import binary_crossentropy
from sklearn.utils import check_random_state


class FETALinearCore(Learner):
    def __init__(self, n_object_features, n_objects, learning_rate=1e-3, batch_size=256,
                 loss_function=binary_crossentropy, epochs_drop=50, drop=0.01, random_state=None, **kwargs):
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
        self.w_out = None
        self.optimizer = None
        self.W_last = None

    def _construct_model_(self, n_objects):
        self.X = tf.placeholder("float32", [None, n_objects, self.n_object_features])
        self.Y = tf.placeholder("float32", [None, n_objects])
        std = 1 / np.sqrt(self.n_object_features)
        self.b1 = tf.Variable(self.random_state.normal(loc=0, scale=std, size=1), dtype=tf.float32)
        self.W1 = tf.Variable(self.random_state.normal(loc=0, scale=std, size=2 * self.n_object_features),
                              dtype=tf.float32)
        self.W2 = tf.Variable(self.random_state.normal(loc=0, scale=std, size=self.n_object_features), dtype=tf.float32)
        self.b2 = tf.Variable(self.random_state.normal(loc=0, scale=std, size=1), dtype=tf.float32)
        self.W_out = tf.Variable(self.random_state.normal(loc=0, scale=std, size=2), dtype=tf.float32, name='W_out')

        outputs = [list() for _ in range(n_objects)]
        for i, j in combinations(range(n_objects), 2):
            x1 = self.X[:, i]
            x2 = self.X[:, j]
            x1x2 = tf.concat((x1, x2), axis=1)
            x2x1 = tf.concat((x2, x1), axis=1)
            n_g = (tf.tensordot(x1x2, self.W1, axes=1) + self.b1)
            n_l = (tf.tensordot(x2x1, self.W1, axes=1) + self.b1)
            outputs[i].append(n_g[:, None])
            outputs[j].append(n_l[:, None])
        outputs = [tf.concat(x, axis=1) for x in outputs]
        outputs = tf.reduce_mean(outputs, axis=-1)
        outputs = tf.transpose(outputs)
        zero_outputs = tf.tensordot(self.X, self.W2, axes=1) + self.b2
        scores = tf.sigmoid(self.W_out[0] * zero_outputs + self.W_out[1] * outputs)
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
            self.W_last = tf_session.run(self.W_out)

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
        outputs = [list() for _ in range(n_objects)]
        for i, j in combinations(range(n_objects), 2):
            x1 = X[:, i]
            x2 = X[:, j]
            x1x2 = np.concatenate((x1, x2), axis=1)
            x2x1 = np.concatenate((x2, x1), axis=1)
            n_g = (np.dot(x1x2, self.weight1) + self.bias1)
            n_l = (np.dot(x2x1, self.weight1) + self.bias1)
            outputs[i].append(n_g)
            outputs[j].append(n_l)
        outputs = np.array(outputs)
        outputs = np.mean(outputs, axis=1).T
        scores_zero = (np.dot(X, self.weight2) + self.bias2)
        scores = sigmoid(self.W_last[0] * scores_zero + self.W_last[1] * outputs)
        return scores

    def set_tunable_parameters(self, learning_rate=1e-3, batch_size=128, epochs_drop=300, drop=0.1, **point):
        """
            Set tunable parameters of the FETA-network to the values provided.

            Parameters
            ----------
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
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self._construct_model_(self.n_objects)
        self.epochs_drop = epochs_drop
        self.drop = drop
        if len(point) > 0:
            self.logger.warning('This ranking algorithm does not support tunable parameters'
                                ' called: {}'.format(print_dictionary(point)))
