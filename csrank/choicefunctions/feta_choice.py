import logging
from itertools import combinations

import numpy as np
from keras import Input, backend as K
from keras.layers import Dense, concatenate, Lambda, add, Activation
from keras.losses import binary_crossentropy
from keras.regularizers import l2
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from csrank.layers import NormalizedDense
from csrank.objectranking import FETAObjectRanker


class FETAChoiceFunction(FETAObjectRanker):
    def __init__(self, n_objects, n_object_features, n_hidden=2, n_units=8,
                 add_zeroth_order_model=False, max_number_of_objects=5,
                 num_subsample=5, loss_function=binary_crossentropy,
                 batch_normalization=False, kernel_regularizer=l2(l=1e-4),
                 non_linearities='selu', optimizer="adam", metrics=None, batch_size=256,
                 random_state=None, **kwargs):
        super().__init__(n_objects, n_object_features, n_hidden, n_units,
                         add_zeroth_order_model, max_number_of_objects,
                         num_subsample, loss_function, batch_normalization,
                         kernel_regularizer, non_linearities, optimizer,
                         metrics, batch_size,
                         random_state, **kwargs)
        self.threshold = 0.5
        self.logger = logging.getLogger(FETAChoiceFunction.__name__)

    def _construct_layers(self, **kwargs):
        self.input_layer = Input(shape=(self.n_objects, self.n_object_features))
        # Todo: Variable sized input
        # X = Input(shape=(None, n_features))
        if self.batch_normalization:
            if self._use_zeroth_model:
                self.hidden_layers_zeroth = [
                    NormalizedDense(self.n_units,
                                    name="hidden_zeroth_{}".format(x),
                                    kernel_regularizer=self.kernel_regularizer,
                                    activation=self.non_linearities,
                                    **kwargs)
                    for x in range(self.n_hidden)
                ]
            self.hidden_layers = [
                NormalizedDense(self.n_units, name="hidden_{}".format(x),
                                kernel_regularizer=self.kernel_regularizer,
                                activation=self.non_linearities,
                                **kwargs)
                for x in range(self.n_hidden)
            ]
        else:
            if self._use_zeroth_model:
                self.hidden_layers_zeroth = [
                    Dense(self.n_units, name="hidden_zeroth_{}".format(x),
                          kernel_regularizer=self.kernel_regularizer,
                          activation=self.non_linearities,
                          **kwargs)
                    for x in range(self.n_hidden)
                ]
            self.hidden_layers = [
                Dense(self.n_units, name="hidden_{}".format(x),
                      kernel_regularizer=self.kernel_regularizer,
                      activation=self.non_linearities,
                      **kwargs)
                for x in range(self.n_hidden)
            ]
        assert len(self.hidden_layers) == self.n_hidden
        self.output_node = Dense(1, activation="linear",
                                 kernel_regularizer=self.kernel_regularizer)
        if self._use_zeroth_model:
            self.output_node_zeroth = Dense(1, activation="linear",
                                            kernel_regularizer=self.kernel_regularizer)

    def _tune_threshold(self, X_val, Y_val, thin_thresholds=1):
        scores = self.predict_scores(X_val)
        probs = np.unique(scores)[::thin_thresholds]
        threshold = 0.0
        best = f1_score(Y_val, scores > threshold, average='samples')
        for i, p in enumerate(probs):
            pred = scores > p
            f1 = f1_score(Y_val, pred, average='samples')
            if f1 > best:
                threshold = p
                best = f1
        self.logger.info('Tuned threshold, obtained {:.2f} which achieved'
                         ' a micro F1-measure of {:.2f}'.format(
            threshold, best))
        return threshold

    def fit(self, X, Y, epochs=10, callbacks=None, validation_split=0.1,
            tune_size=0.1, thin_thresholds=1, verbose=0, **kwd):
        if tune_size > 0:
            X_train, X_val, Y_train, Y_val = train_test_split(
                X, Y, test_size=tune_size)
            try:
                super().fit(X_train, Y_train, epochs, callbacks,
                            validation_split, verbose, **kwd)
            finally:
                self.logger.info('Fitting utility function finished. Start tuning threshold.')
                self.threshold = self._tune_threshold(
                    X_val, Y_val,
                    thin_thresholds=thin_thresholds)
        else:
            super().fit(X, Y, epochs, callbacks, validation_split, verbose,
                        **kwd)
            self.threshold = 0.5

    def predict_scores(self, X, **kwargs):
        return super().predict_scores(X, **kwargs)

    def predict(self, X, **kwargs):
        self.logger.debug('Predicting started')

        scores = self.predict_scores(X, **kwargs)
        self.logger.debug('Predicting scores complete')
        if isinstance(X, dict):
            result = dict()
            for n, s in self.predict_scores(X, **kwargs).items():
                result[n] = s > self.threshold
        else:
            result = scores > self.threshold
        return result

    def __call__(self, X, **kwargs):
        return self.predict(X, **kwargs)

    def sub_sampling(self, X, Y):
        if self._n_objects <= self.max_number_of_objects:
            return X, Y
        n_objects = self.max_number_of_objects
        bucket_size = 1  # int(X.shape[1] / n_objects) + 2
        X_train = []
        Y_train = []
        for x, y in zip(X, Y):
            ind_1 = np.where(y == 1)[0]
            p_1 = np.zeros(len(ind_1)) + 1 / len(ind_1)
            if (y == 1).sum() < n_objects:
                ind_0 = np.where(y == 0)[0]
                p_0 = np.zeros(len(ind_0)) + 1 / len(ind_0)
                positives = (y == 1).sum() if n_objects > (
                        y == 1).sum() else n_objects
                if positives > bucket_size:
                    cp = self.random_state.choice(positives, size=bucket_size,
                                                  replace=False) + 1
                else:
                    cp = self.random_state.choice(positives,
                                                  size=bucket_size) + 1
                idx = []
                for c in cp:
                    pos = self.random_state.choice(len(ind_1), size=c,
                                                   replace=False, p=p_1)
                    if n_objects - c > len(ind_0):
                        neg = self.random_state.choice(len(ind_0),
                                                       size=n_objects - c,
                                                       p=p_0)
                    else:
                        neg = self.random_state.choice(len(ind_0),
                                                       size=n_objects - c,
                                                       replace=False, p=p_0)
                    p_0[neg] = 0.2 * p_0[neg]
                    p_0 = p_0 / p_0.sum()
                    i = np.concatenate((ind_1[pos], ind_0[neg]))
                    self.random_state.shuffle(i)
                    p_1[pos] = 0.2 * p_1[pos]
                    p_1 = p_1 / p_1.sum()
                    p_0[neg] = 0.2 * p_0[neg]
                    p_0 = p_0 / p_0.sum()
                    idx.append(i)
                idx = np.array(idx)
            else:
                idx = self.random_state.choice(ind_1,
                                               size=(bucket_size, n_objects))
                idx = np.array(idx)
            if len(X_train) == 0:
                X_train = x[idx]
                Y_train = y[idx]
            else:
                Y_train = np.concatenate([Y_train, y[idx]], axis=0)
                X_train = np.concatenate([X_train, x[idx]], axis=0)
        print("Sampled instances {} objects {}".format(X_train.shape[0],
                                                       X_train.shape[1]))
        return X_train, Y_train
    def construct_model(self):
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
            scores = add([scores, zeroth_order_scores])
        return Activation('sigmoid')(scores)
