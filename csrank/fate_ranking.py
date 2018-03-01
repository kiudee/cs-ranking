import logging
from abc import ABCMeta, abstractmethod
from collections import OrderedDict

import keras.backend as K
import numpy as np
from keras import optimizers
from keras.layers import Input, Dense
from keras.layers.merge import concatenate
from keras.models import Model
from keras.regularizers import l2
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import check_random_state

from csrank.callbacks import EarlyStoppingWithWeights, LRScheduler
from csrank.constants import BATCH_SIZE, BATCH_SIZE_DEFAULT_RANGE, \
    LEARNING_RATE, LR_DEFAULT_RANGE, \
    REGULARIZATION_FACTOR, \
    REGULARIZATION_FACTOR_DEFAULT_RANGE, \
    EARLY_STOPPING_PATIENCE_DEFAULT_RANGE, \
    EARLY_STOPPING_PATIENCE
from csrank.discretechoice.discrete_choice import ObjectChooser
from csrank.dyadranking.contextual_ranking import ContextualRanker
from csrank.labelranking.label_ranker import LabelRanker
from csrank.layers import DeepSet
from csrank.losses import hinged_rank_loss, smooth_rank_loss
from csrank.metrics import zero_one_rank_loss_for_scores_ties, \
    zero_one_rank_loss_for_scores
from csrank.objectranking.object_ranker import ObjectRanker
from csrank.tunable import Tunable
from csrank.util import scores_to_rankings, create_input_lambda, \
    tunable_parameters_ranges, tensorify, \
    print_dictionary, deprecated

GENERAL_OBJECT_CHOOSER = "FATEObjectChooser"
GENERAL_LABEL_RANKER = "FATELabelRanker"
GENERAL_OBJECT_RANKER = "FATEObjectRanker"
GENERAL_RANKING_CORE = "FATERankingCore"
GENERAL_OBJECT_RANKING_CORE = "FATEObjectRankingCore"

N_HIDDEN_SET_LAYERS = "n_hidden_set_layers"
N_HIDDEN_SET_UNITS = "n_hidden_set_units"
N_HIDDEN_JOINT_UNITS = 'n_hidden_joint_units'
N_HIDDEN_JOINT_LAYERS = 'n_hidden_joint_layers'
SET_LAYERS_DEFAULT_RANGES = (1, 20)
SET_HIDDEN_UNITS_DEFAULT_RANGE = (8, 256)
JOINT_HIDDEN_LAYERS_DEFAULT_RANGE = (1, 20)
JOINT_HIDDEN_UNITS_DEFAULT_RANGE = (8, 256)

__all__ = ['FATELabelRanker', 'FATEObjectRanker', 'FATEContextualRanker', 'FATEObjectChooser']


class FATERankingCore(Tunable, metaclass=ABCMeta):
    _tunable = None
    _use_early_stopping = None

    def __init__(self, n_hidden_joint_layers=32, n_hidden_joint_units=32,
                 activation='selu', kernel_initializer='lecun_normal',
                 kernel_regularizer=l2(l=0.01),
                 optimizer="adam",
                 es_patience=300, use_early_stopping=False, batch_size=256,
                 random_state=None, **kwargs):
        self.logger = logging.getLogger(GENERAL_RANKING_CORE)
        self.random_state = check_random_state(random_state)

        self.n_hidden_joint_layers = n_hidden_joint_layers
        self.n_hidden_joint_units = n_hidden_joint_units

        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        self.batch_size = batch_size
        self.optimizer = optimizers.get(optimizer)
        self.early_stopping = EarlyStoppingWithWeights(patience=es_patience)
        self._use_early_stopping = use_early_stopping
        self.__kwargs__ = kwargs
        self._construct_layers(activation=self.activation,
                               kernel_initializer=self.kernel_initializer,
                               kernel_regularizer=self.kernel_regularizer)

    def _construct_layers(self, **kwargs):
        """ Construct basic layers shared by all ranking algorithms:
         * Joint dense hidden layers
         * Output scoring layer

        Connecting the layers is done in join_input_layers and will be done in
        implementing classes.
        """
        self.logger.info(
            "Construct joint layers hidden units {} and layers {} ".format(
                self.n_hidden_joint_units,
                self.n_hidden_joint_layers))
        # Create joint hidden layers:
        self.joint_layers = []
        for i in range(self.n_hidden_joint_layers):
            self.joint_layers.append(
                Dense(self.n_hidden_joint_units,
                      name="joint_layer_{}".format(i),
                      **kwargs)
            )

        self.logger.info('Construct output score node')
        self.scorer = Dense(1, name="output_node", activation='linear',
                            kernel_regularizer=self.kernel_regularizer)

    def join_input_layers(self, input_layer, *layers, n_layers, n_objects):
        """
        Accepts input tensors and an arbitrary number of feature tensors
        and concatenates them into a joint layer.
        The input layers need to be given separately, because they need to be
        iterated over.

        Parameters
        ----------
        input_layer : input tensor (n_objects, n_features)
        layers : tensors
            A number of tensors representing feature representations
        n_layers : int
            Number of joint hidden layers
        n_objects : int
            Number of objects
        """
        self.logger.debug("Joining set representation and joint layers")
        scores = []

        inputs = [create_input_lambda(i)(input_layer) for i in
                  range(n_objects)]

        for i in range(n_objects):
            if n_layers >= 1:
                joint = concatenate([inputs[i], *layers])
            else:
                joint = inputs[i]
            for j in range(len(self.joint_layers)):
                joint = self.joint_layers[j](joint)
            scores.append(self.scorer(joint))
        scores = concatenate(scores, name="final_scores")
        self.logger.debug("Done")

        return scores

    def set_tunable_parameters(self, point):
        named = Tunable.set_tunable_parameters(self, point)
        hidden_layers_created = False

        LAYER_KEYS = [N_HIDDEN_JOINT_UNITS, N_HIDDEN_JOINT_LAYERS]
        del named[N_HIDDEN_SET_UNITS]
        del named[N_HIDDEN_SET_LAYERS]

        for name, param in named.items():
            if name in LAYER_KEYS + [REGULARIZATION_FACTOR]:
                selected_params = {k: v for k, v in named.items()
                                   if k in LAYER_KEYS}
                self.__dict__.update(selected_params)
                self.kernel_regularizer = l2(l=named[REGULARIZATION_FACTOR])
                if not hidden_layers_created:
                    self._construct_layers(
                        activation=self.activation,
                        kernel_initializer=self.kernel_initializer,
                        kernel_regularizer=self.kernel_regularizer)
                hidden_layers_created = True
            elif name == LEARNING_RATE:
                K.set_value(self.optimizer.lr, param)
            elif name == EARLY_STOPPING_PATIENCE:
                self.early_stopping.patience = param
            elif name == BATCH_SIZE:
                self.batch_size = param
            else:
                self.logger.warning('This ranking algorithm does not support'
                                    ' a tunable parameter'
                                    ' called {}'.format(name))

    @classmethod
    def tunable_parameters(cls):
        logger = logging.getLogger(GENERAL_RANKING_CORE)
        logger.info("Tunable parameters")
        if cls._tunable is None:
            logger.info("tunable is None")
            cls._tunable = dict([
                (N_HIDDEN_JOINT_UNITS, JOINT_HIDDEN_UNITS_DEFAULT_RANGE),
                (N_HIDDEN_JOINT_LAYERS, JOINT_HIDDEN_LAYERS_DEFAULT_RANGE),
                (LEARNING_RATE, LR_DEFAULT_RANGE),
                (REGULARIZATION_FACTOR, REGULARIZATION_FACTOR_DEFAULT_RANGE),
                (BATCH_SIZE, BATCH_SIZE_DEFAULT_RANGE),
            ])
        logger.info("Core ranking params {}".format(print_dictionary(cls._tunable)))

    @abstractmethod
    def fit(self, *args, **kwargs):
        """Fit the general ranking algorithm. The implementing classes need to
        handle the feature information and output representation.

        Returns
        -------
        self : object
            Returns self.
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, *args, **kwargs):
        """Predict a suitable ranking output for the given ranking problem."""
        raise NotImplementedError


class FATEObjectRankingCore(FATERankingCore, metaclass=ABCMeta):
    def __init__(self, n_object_features,
                 n_hidden_set_layers=1, n_hidden_set_units=1,
                 **kwargs):
        FATERankingCore.__init__(self, **kwargs)
        self.logger_gorc = logging.getLogger(GENERAL_OBJECT_RANKING_CORE)

        self.n_hidden_set_layers = n_hidden_set_layers
        self.n_hidden_set_units = n_hidden_set_units
        self.n_object_features = n_object_features
        self.logger_gorc.info("args: {}".format(repr(kwargs)))
        self._create_set_layers(activation=self.activation,
                                kernel_initializer=self.kernel_initializer,
                                kernel_regularizer=self.kernel_regularizer)
        self.is_variadic = True

    def _create_set_layers(self, **kwargs):
        """Create layers for learning the representation of the query set.

        The actual connection of the layers is done during fitting, since we
        do not know the size(s) of the set(s) in advance."""
        self.logger_gorc = logging.getLogger(GENERAL_OBJECT_RANKING_CORE)
        self.logger_gorc.info("Creating set layers with set units {} "
                              "set layer {} ".format(
            self.n_hidden_set_units,
            self.n_hidden_set_layers))

        if self.n_hidden_set_layers >= 1:
            self.set_layer = DeepSet(units=self.n_hidden_set_units,
                                     layers=self.n_hidden_set_layers,
                                     **kwargs)
        else:
            self.set_layer = None

    @deprecated
    def _construct_scoring_model(self, n_objects_test, X, **kwargs):
        """
        Construct a scoring model for prediction on the given number of objects.

        The existing layers of the networks are used during the construction
        and the weights are shared.

        Parameters
        ----------
        n_objects_test : int
            Number of objects for which to compute scores
        """
        input_layer_scorer = Input(shape=(n_objects_test,
                                          self.n_object_features),
                                   name="input_node")
        if self.n_hidden_set_layers >= 1:
            set_repr = self.set_layer(input_layer_scorer)
        else:
            set_repr = None

        scores = self.join_input_layers(input_layer_scorer, set_repr,
                                        n_layers=self.n_hidden_set_layers,
                                        n_objects=n_objects_test)
        scoring_model = Model(inputs=input_layer_scorer, outputs=scores)
        return scoring_model

    @staticmethod
    def _bucket_frequencies(X, min_bucket_size=32):
        """
        Calculates the relative frequency of each ranking bucket.

        Parameters
        ----------
        X : dict
            map from n_objects to object queries
        min_bucket_size : int
            Minimum number of instances for a query size to be considered for
            the frequency calculation

        Returns
        -------
        freq : dict
            map from n_objects to frequency in float

        """
        freq = dict()
        total = 0.0
        for n_objects, arr in X.items():
            n_instances = arr.shape[0]
            if n_instances >= min_bucket_size:
                freq[n_objects] = n_instances
                total += freq[n_objects]
            else:
                freq[n_objects] = 0
        for n_objects in freq.keys():
            freq[n_objects] /= total
        return freq

    def _construct_models(self, buckets):
        models = dict()
        n_features = self.n_object_features

        for n_objects in buckets.keys():
            input_layer = Input(shape=(n_objects, n_features),
                                name="input_node")

            set_repr = self.set_layer(input_layer)

            scores = self.join_input_layers(input_layer, set_repr,
                                            n_objects=n_objects,
                                            n_layers=self.n_hidden_set_layers)
            model = Model(inputs=input_layer, outputs=scores)
            model.compile(loss=self.loss_function,
                          optimizer=self.optimizer,
                          metrics=self.metrics)
            models[n_objects] = model
        return models

    def get_weights(self):
        if self.is_variadic:
            return list(self.models_.values())[0].get_weights()
        else:
            return self.model.get_weights()

    def set_weights(self, weights):
        if self.is_variadic:
            list(self.models_.values())[0].set_weights(weights)
        else:
            self.model.set_weights(weights)

    def _fit(self, X=None, Y=None, generator=None, epochs=35, inner_epochs=1,
             log_callbacks=None, validation_split=0.1, verbose=0, global_lr=1.0,
             global_momentum=0.9, min_bucket_size=500, refit=False, **kwargs):
        if isinstance(X, dict):
            if generator is not None:
                self.logger.error("Variadic training does not support"
                                  " generators yet.")
                raise NotImplementedError
            self.is_variadic = True
            decay_rate = global_lr / epochs
            learning_rate = global_lr
            freq = self._bucket_frequencies(X, min_bucket_size=min_bucket_size)
            bucket_ids = np.array(tuple(X.keys()))

            #  Create models which need to be trained
            #  Note, that the models share all their weights, the only
            #  difference is the compute graph constructed for back propagation.
            if not hasattr(self, 'models_') or refit:
                self.models_ = self._construct_models(X)

            #  Iterate training
            for epoch in range(epochs):

                self.logger.info("Epoch: {}, Learning rate: {}"
                                 .format(epoch, learning_rate))

                # In the spirit of mini-batch SGD we also shuffle the buckets
                # each epoch:
                np.random.shuffle(bucket_ids)

                w_before = np.array(self.get_weights())

                for bucket_id in bucket_ids:

                    # Skip query sizes with too few instances:
                    if X[bucket_id].shape[0] < min_bucket_size:
                        continue

                    # self.set_weights(start)
                    x = X[bucket_id]
                    y = Y[bucket_id]
                    # TODO: How to do callbacks fit here?

                    # Save weight vector for momentum:
                    w_old = w_before
                    w_before = np.array(self.get_weights())
                    self.models_[bucket_id].fit(
                        x=x, y=y,
                        epochs=inner_epochs,
                        batch_size=self.batch_size,
                        validation_split=validation_split,
                        verbose=verbose,
                        **kwargs
                    )
                    w_after = np.array(self.get_weights())
                    self.set_weights(w_before
                                     + learning_rate * freq[bucket_id]
                                     * (w_after - w_before)
                                     + global_momentum * (w_before - w_old))
                learning_rate /= 1 + decay_rate * epoch
        else:
            self.is_variadic = False

            if not self.model is not None or refit:
                if generator is not None:
                    X, Y = next(iter(generator))

                n_inst, n_objects, n_features = X.shape

                input_layer = Input(shape=(n_objects,
                                           n_features),
                                    name="input_node")

                set_repr = self.set_layer(input_layer)
                scores = self.join_input_layers(input_layer, set_repr,
                                                n_objects=n_objects,
                                                n_layers=self.n_hidden_set_layers)
                self.model = Model(inputs=input_layer, outputs=scores)
            self.model.compile(loss=self.loss_function,
                                optimizer=self.optimizer,
                                 metrics=self.metrics)
            callbacks = []
            if log_callbacks is None:
                log_callbacks = []
            callbacks.extend(log_callbacks)
            for c in callbacks:
                if isinstance(c, LRScheduler):
                    c.initial_lr = K.get_value(self.optimizer.lr)
                    self.logger.info("Setting lr {} for {}".format(c.initial_lr, c.__name__))
            if self._use_early_stopping:
                callbacks.append(self.early_stopping)

            self.logger.info("Callbacks {}".format(
                ', '.join([c.__name__ for c in callbacks])))

            self.logger.info("Fitting started")
            if generator is None:
                self.model.fit(
                    x=X, y=Y, callbacks=callbacks, epochs=epochs,
                    validation_split=validation_split, batch_size=self.batch_size,
                    verbose=verbose, **kwargs)
            else:
                self.model.fit_generator(
                    generator=generator, callbacks=callbacks, epochs=epochs,
                    # validation_split=validation_split, # TODO: fix
                    verbose=verbose, **kwargs)
            self.logger.info("Fitting complete")

    def fit(self, X, Y, epochs=35, inner_epochs=1, log_callbacks=None,
            validation_split=0.1, verbose=0,
            global_lr=1.0, global_momentum=0.9,
            min_bucket_size=500, refit=False, **kwargs):
        """Fit a generic object ranking model on a provided set of queries.

        The provided queries can be of a fixed size (numpy arrays) or of
        varying sizes in which case dictionaries are expected as input.

        For varying sizes a meta gradient descent is performed across the
        different query sizes.

        Parameters
        ----------
        X : numpy array or dict
            (n_instances, n_objects, n_features) if numpy array or
            map from n_objects to numpy arrays
        Y : numpy array or dict
            (n_instances, n_objects) if numpy array or
            map from n_objects to numpy arrays
        epochs : int
            Number of epochs to run if training for a fixed query size or
            number of epochs of the meta gradient descent for the variadic model
        inner_epochs : int
            Number of epochs to train for each query size inside the variadic
            model
        log_callbacks : list
            List of callbacks to be called during optimization
        validation_split : float
            Percentage of instances to split off to validate on
        verbose : bool
            Print verbose information
        global_lr : float
            Learning rate of the meta gradient descent (variadic model only)
        global_momentum : float
            Momentum for the meta gradient descent (variadic model only)
        min_bucket_size : int
            Restrict the training to queries of a minimum size
        refit : bool
            If True, create a new model object, otherwise continue fitting the
            existing one if one exists.
        """
        self._fit(X=X, Y=Y, epochs=epochs, inner_epochs=inner_epochs,
                  log_callbacks=log_callbacks,
                  validation_split=validation_split, verbose=verbose,
                  global_lr=global_lr, global_momentum=global_momentum,
                  min_bucket_size=min_bucket_size, refit=refit, **kwargs)

    def fit_generator(self, generator, epochs=35, inner_epochs=1,
                      log_callbacks=None, validation_split=0.1, verbose=0,
                      global_lr=1.0, global_momentum=0.9, min_bucket_size=500,
                      refit=False, **kwargs):
        """Fit a generic object ranking model on a set of queries provided by
        a generator.

        The provided queries can be of a fixed size (numpy arrays) or of
        varying sizes in which case dictionaries are expected as input.

        For varying sizes a meta gradient descent is performed across the
        different query sizes.

        Parameters
        ----------
        X : numpy array or dict
            (n_instances, n_objects, n_features) if numpy array or
            map from n_objects to numpy arrays
        Y : numpy array or dict
            (n_instances, n_objects) if numpy array or
            map from n_objects to numpy arrays
        epochs : int
            Number of epochs to run if training for a fixed query size or
            number of epochs of the meta gradient descent for the variadic model
        inner_epochs : int
            Number of epochs to train for each query size inside the variadic
            model
        log_callbacks : list
            List of callbacks to be called during optimization
        validation_split : float
            Percentage of instances to split off to validate on
        verbose : bool
            Print verbose information
        global_lr : float
            Learning rate of the meta gradient descent (variadic model only)
        global_momentum : float
            Momentum for the meta gradient descent (variadic model only)
        min_bucket_size : int
            Restrict the training to queries of a minimum size
        refit : bool
            If True, create a new model object, otherwise continue fitting the
            existing one if one exists.
        """
        self._fit(generator=generator, epochs=epochs, inner_epochs=inner_epochs,
                  log_callbacks=log_callbacks,
                  validation_split=validation_split, verbose=verbose,
                  global_lr=global_lr, global_momentum=global_momentum,
                  min_bucket_size=min_bucket_size, refit=refit, **kwargs)

    def get_set_representaion(self, X, kwargs):
        n_instances, n_objects, n_features = X.shape
        self.logger.info("Test Set instances {} objects {} features {}".format(n_instances, n_objects, n_features))
        input_layer_scorer = Input(shape=(n_objects,
                                          self.n_object_features),
                                   name="input_node")
        if self.n_hidden_set_layers >= 1:
            self.set_layer(input_layer_scorer)
            fr = self.set_layer.cached_models[n_objects].predict(X, **kwargs)
            del self.set_layer.cached_models[n_objects]
            X_n = np.empty((fr.shape[0], n_objects, fr.shape[1] + self.n_object_features), dtype="float")
            for i in range(n_objects):
                X_n[:, i] = np.concatenate((X[:, i], fr), axis=1)
            X = np.copy(X_n)
        return X

    def _predict_scores_fixed(self, X, **kwargs):
        """
        Predict the scores for a fixed ranking size.

        Parameters
        ----------
        X : numpy array
            float (n_instances, n_objects, n_features)

        Returns
        -------
        scores : numpy array
            float (n_instances, n_objects)

        """
        # model = self._construct_scoring_model(n_objects)
        X = self.get_set_representaion(X, kwargs)
        n_instances, n_objects, n_features = X.shape
        self.logger.info(
            "After applying the set representations instances {} objects {} features {}".format(n_instances, n_objects,
                                                                                                n_features))
        input_layer_joint = Input(shape=(n_objects, n_features), name="input_joint_model")
        scores = []

        inputs = [create_input_lambda(i)(input_layer_joint) for i in
                  range(n_objects)]

        for i in range(n_objects):
            joint = inputs[i]
            for j in range(len(self.joint_layers)):
                joint = self.joint_layers[j](joint)
            scores.append(self.scorer(joint))
        scores = concatenate(scores, name="final_scores")
        joint_model = Model(inputs=input_layer_joint, outputs=scores)
        predicted_scores = joint_model.predict(X)
        self.logger.info("Done predicting scores")
        return predicted_scores

    def predict_scores(self, X, **kwargs):
        """
        Predict the latent utility scores for each object in X.

        We need to distinguish several cases here:
         * Predict with the non-variadic model on the same ranking size
         * Predict with the non-variadic model on a new ranking size
         * Predict with the variadic model on a known ranking size
         * Predict with the variadic model on a new ranking size
         * Predict on a variadic input

        The simplest solution is creating (a) new model(s) in all of the cases,
        even though it/they might already exist.

         Parameters
         ----------
         X : dict or numpy array
            Dictionary with a mapping from ranking size to numpy arrays
            or a single numpy array of size:
            (n_instances, n_objects, n_features)
         """
        self.logger.info("Predicting scores")

        if isinstance(X, dict):
            scores = dict()
            for ranking_size, x in X.items():
                scores[ranking_size] = self._predict_scores_fixed(x, **kwargs)
        else:
            scores = self._predict_scores_fixed(X, **kwargs)
        return scores

    def set_tunable_parameters(self, point):
        FATERankingCore.set_tunable_parameters(self, point)
        hidden_layers_created = False
        self.logger_gorc = logging.getLogger(GENERAL_OBJECT_RANKING_CORE)
        keys = self._tunable.copy().keys()
        named = dict(zip(keys, point))

        SET_LAYER_KEYS = [N_HIDDEN_SET_UNITS, N_HIDDEN_SET_LAYERS]
        for key in keys:
            if key not in SET_LAYER_KEYS + [REGULARIZATION_FACTOR]:
                del named[key]

        for name, param in named.items():
            if name in SET_LAYER_KEYS + [REGULARIZATION_FACTOR]:
                selected_params = {k: v for k, v in named.items()
                                   if k in SET_LAYER_KEYS}
                self.__dict__.update(selected_params)
                self.kernel_regularizer = l2(l=named[REGULARIZATION_FACTOR])
                if not hidden_layers_created:
                    self._create_set_layers(
                        activation=self.activation,
                        kernel_initializer=self.kernel_initializer,
                        kernel_regularizer=self.kernel_regularizer)
                hidden_layers_created = True
            else:
                self.logger_gorc.warning(
                    "This ranking algorithm does not support a tunable parameter called {}".format(name))

    @classmethod
    def tunable_parameters(cls):
        logger = logging.getLogger(GENERAL_OBJECT_RANKING_CORE)
        if cls._tunable is None:
            FATERankingCore.tunable_parameters()
        logger.info("Use Early Stopping {}".format(cls._use_early_stopping))
        if cls._use_early_stopping:
            cls._tunable[EARLY_STOPPING_PATIENCE] = EARLY_STOPPING_PATIENCE_DEFAULT_RANGE
        dict_params = dict([
            (N_HIDDEN_SET_UNITS, SET_HIDDEN_UNITS_DEFAULT_RANGE),
            (N_HIDDEN_SET_LAYERS, SET_LAYERS_DEFAULT_RANGES),
            (REGULARIZATION_FACTOR, REGULARIZATION_FACTOR_DEFAULT_RANGE)])
        logger.info("Set params {}".format(print_dictionary(dict_params)))
        for k, v in dict_params.items():
            cls._tunable[k] = v
        cls._tunable = OrderedDict(sorted(cls._tunable.items()))

    @classmethod
    def set_tunable_parameter_ranges(cls, param_ranges_dict):
        logger = logging.getLogger(GENERAL_OBJECT_RANKING_CORE)
        return tunable_parameters_ranges(cls, logger, param_ranges_dict)


class FATEObjectRanker(FATEObjectRankingCore, ObjectRanker):
    """ Create a FATE-network architecture for object ranking.

        Training complexity is quadratic in the number of objects and
        prediction complexity is only linear.

        Parameters
        ----------
        n_object_features : int
            Dimensionality of the feature space of each object
        n_hidden_set_layers : int
            Number of hidden layers for the context representation
        n_hidden_set_units : int
            Number of hidden units in each layer of the context representation
        loss_function : function
            Differentiable loss function for the score vector
        metrics : list
            List of evaluation metrics (can be non-differentiable)
        **kwargs
            Keyword arguments for the hidden units
        """

    def __init__(self, n_object_features,
                 n_hidden_set_layers=2,
                 n_hidden_set_units=32,
                 loss_function=smooth_rank_loss,
                 metrics=None,
                 **kwargs):
        FATEObjectRankingCore.__init__(self,
                                       n_object_features=n_object_features,
                                       n_hidden_set_layers=n_hidden_set_layers,
                                       n_hidden_set_units=n_hidden_set_units,
                                       **kwargs)
        self.loss_function = loss_function
        self.logger = logging.getLogger(GENERAL_OBJECT_RANKER)
        if metrics is None:
            metrics = [zero_one_rank_loss_for_scores_ties,
                       zero_one_rank_loss_for_scores]
        self.metrics = metrics
        self.model = None
        self.logger.info("Initializing network with object features {}".format(
            self.n_object_features))

    def predict(self, X, **kwargs):
        self.logger.info("Predicting ranks")
        if isinstance(X, dict):
            result = dict()
            for n, scores in self.predict_scores(X, **kwargs).items():
                predicted_rankings = scores_to_rankings(scores)
                result[n] = predicted_rankings
            return result
        return ObjectRanker.predict(self, X, **kwargs)

    def _predict_scores_fixed(self, X, **kwargs):
        return FATEObjectRankingCore._predict_scores_fixed(self, X, **kwargs)


class FATEObjectChooser(FATEObjectRankingCore, ObjectChooser):
    def __init__(self, loss_function='categorical_hinge', metrics=None,
                 **kwargs):
        FATEObjectRankingCore.__init__(self, **kwargs)
        self.loss_function = loss_function
        if metrics is None:
            metrics = ['categorical_accuracy']
        self.metrics = metrics
        self.model = None
        self.logger = logging.getLogger(GENERAL_OBJECT_CHOOSER)

    def predict(self, X, **kwargs):
        scores = self.predict_scores(X, **kwargs)
        if self.is_variadic:
            result = dict()
            for n, s in scores.items():
                result[n] = s.argmax(axis=1)
        else:
            self.logger.info("Predicting chosen object")
            result = scores.argmax(axis=1)
        return result


class FATELabelRanker(FATERankingCore, LabelRanker):
    def __init__(self, loss_function=hinged_rank_loss, metrics=None,
                 **kwargs):
        super().__init__(self, label_ranker=True, **kwargs)
        self.loss_function = loss_function
        self.logger = logging.getLogger(GENERAL_LABEL_RANKER)
        if metrics is None:
            metrics = [zero_one_rank_loss_for_scores_ties,
                       zero_one_rank_loss_for_scores]
        self.metrics = metrics
        self.model = None
        self.logger.info("Initializing network with object features {}".format(
            self.n_object_features))
        self._connect_layers()

    def one_hot_encoder_lr_data_conversion(self, X, Y):
        X_trans = []
        for i, x in enumerate(X):
            x = x[None, :]
            x = np.repeat(x, len(Y[i]), axis=0)
            label_binarizer = LabelBinarizer()
            label_binarizer.fit(range(max(Y[i]) + 1))
            b = label_binarizer.transform(Y[i])
            x = np.concatenate((x, b), axis=1)
            X_trans.append(x)
        X_trans = np.array(X_trans)
        return X_trans

    def _create_set_layers(self, **kwargs):
        FATEObjectRankingCore._create_set_layers(self, **kwargs)

    def _connect_layers(self):
        self.set_input_layers(self.inputs, self.set_repr,
                              self.n_hidden_set_layers)

    def fit(self, X, Y, log_callbacks=None, validation_split=0.1, verbose=0,
            **kwargs):
        self.logger.info("Fitting started")
        X_trans = self.one_hot_encoder_lr_data_conversion(X, Y)

        self.model = Model(inputs=self.input_layer, outputs=self.scores)
        self.model.compile(loss=self.loss_function, optimizer=self.optimizer,
                           metrics=self.metrics)
        callbacks = []
        if log_callbacks is None:
            log_callbacks = []
        callbacks.extend(log_callbacks)
        if self._use_early_stopping:
            callbacks.append(self.early_stopping)

        self.logger.info("Callbacks {}".format(
            ', '.join([c.__name__ for c in callbacks])))

        self.model.fit(
            x=X_trans, y=Y, callbacks=callbacks,
            validation_split=validation_split,
            batch_size=self.batch_size,
            verbose=verbose, **kwargs)
        self.logger.info("Fitting completed")

    def predict_scores(self, X, **kwargs):
        self.logger.info("Predicting scores")
        n_instances, n_objects, n_features = tensorify(X).get_shape().as_list()
        Y = []
        for i in range(n_instances):
            Y.append(np.arange(n_objects))
        Y = np.array(Y)
        X_trans = self.one_hot_encoder_lr_data_conversion(X, Y)
        return self.model.predict(X_trans, **kwargs)

    def predict(self, X, **kwargs):
        self.logger.info("Predicting ranks")
        return LabelRanker.predict(self, X, **kwargs)


class FATEContextualRanker(FATEObjectRankingCore, ContextualRanker):
    def fit(self, Xo, Xc, Y, **kwargs):
        pass

    def predict_scores(self, Xo, Xc, **kwargs):
        return self.model.predict([Xo, Xc], **kwargs)

    def predict(self, Xo, Xc, **kwargs):
        s = self.predict_scores(Xo, Xc, **kwargs)
        return scores_to_rankings(s)
