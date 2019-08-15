import copy
import logging
import traceback
from datetime import datetime

import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.metrics import categorical_accuracy
from sklearn.model_selection import ShuffleSplit
from sklearn.utils import check_random_state
from skopt import Optimizer
from skopt.space import check_dimension, Categorical, Integer
from skopt.utils import cook_estimator, normalize_dimensions, dump, load

from csrank.constants import OBJECT_RANKING, LABEL_RANKING, DISCRETE_CHOICE, \
    DYAD_RANKING, CHOICE_FUNCTION
from csrank.learner import Learner
from csrank.metrics import *
from csrank.metrics_np import *
from csrank.tensorflow_util import get_mean_loss
from csrank.tunable import Tunable
from csrank.util import duration_till_now, create_dir_recursively, \
    seconds_to_time, \
    convert_to_loss

PARAMETER_OPTIMIZER = "ParameterOptimizer"

accuracy_scores = [f1_measure, precision, recall, instance_informedness, zero_one_accuracy,
                   zero_one_accuracy_for_scores_np,
                   zero_one_accuracy_for_scores, kendalls_tau_for_scores_np, spearman_correlation_for_scores_scipy,
                   kendalls_tau_for_scores,
                   spearman_correlation_for_scores, make_ndcg_at_k_loss, categorical_accuracy_np,
                   topk_categorical_accuracy_np, topk_categorical_accuracy, categorical_accuracy]


class TuningCallback(object):
    def __init__(self):
        pass

    def set_optimizer(self, opt):
        self.optimizer = opt

    def on_optimization_begin(self, logs=None):
        pass

    def on_optimization_end(self, logs=None):
        pass

    def on_iteration_begin(self, i, logs=None):
        pass

    def on_iteration_end(self, i, logs=None):
        pass


class ParameterOptimizer(Learner):
    def __init__(self, learner, optimizer_path, tunable_parameter_ranges, fit_params=None, random_state=None,
                 tuning_callbacks=None, validation_loss=None, learning_problem=OBJECT_RANKING, **kwd):
        """

        Parameters
        ----------
        learner : object
            The ranker object for which the hyper parameters needs to optimized.
        optimizer_path : string
            The path where the complete optimizer pickle object can be stored.
        tunable_parameter_ranges : dict
            Dictionary with keys as the object needs to be tuned. Values is the dictionary of tunable parameters with ranges in which they should be tuned.
        fit_params : dict
            The fitting parameters for the ranker
        random_state : int or object
            Numpy random state
        tuning_callbacks : list of object
            Callbacks for the parameter optimizer
        validation_loss : function
            Differentiable loss function for the ranker
        learning_problem : string
            The learning problem under which the ranker comes
        **kwd :
            Keyword arguments for the hidden units
        """
        self.logger = logging.getLogger(PARAMETER_OPTIMIZER)

        create_dir_recursively(optimizer_path, True)
        self.optimizer_path = optimizer_path

        self._tunable_parameter_ranges = tunable_parameter_ranges

        check_learner_class(learner)
        self.learner = learner

        if tuning_callbacks is None:
            self.tuning_callbacks = []
        else:
            self.tuning_callbacks = tuning_callbacks
        loss_funcs = {OBJECT_RANKING: zero_one_rank_loss, LABEL_RANKING: zero_one_rank_loss,
                      DISCRETE_CHOICE: categorical_accuracy, DYAD_RANKING: zero_one_rank_loss,
                      CHOICE_FUNCTION: f1_measure}
        if validation_loss is None:
            self.validation_loss = loss_funcs[learning_problem]
            self.logger.info('Loss function is not specified, using {}'.format(loss_funcs[learning_problem].__name__))
        else:
            self.validation_loss = validation_loss

        if self.validation_loss in accuracy_scores:
            self.validation_loss = convert_to_loss(self.validation_loss)
            self.logger.info('Loss function specified is an accuracy converting '
                             '{}'.format(self.validation_loss.__name__))

        if fit_params is None:
            self._fit_params = {}
            self.logger.warning("Fit params are empty, the default parameters will be applied")
        else:
            self._fit_params = fit_params

        self.random_state = check_random_state(random_state)
        self.model = None
        self.opt = None

    def _callbacks_set_optimizer(self, opt):
        for cb in self.tuning_callbacks:
            cb.set_optimizer(opt)

    def _callbacks_on_optimization_begin(self):
        self.logger.info('optimizer begin')
        for cb in self.tuning_callbacks:
            cb.on_optimization_begin()

    def _callbacks_on_optimization_end(self):
        self.logger.info('optimizer end')
        for cb in self.tuning_callbacks:
            cb.on_optimization_end()

    def _callbacks_on_iteration_begin(self, t):
        self.logger.info(
            "********************** optimizer iteration begin ****************"
            "****** " + repr(t))
        for cb in self.tuning_callbacks:
            cb.on_iteration_begin(t)

    def _callbacks_on_iteration_end(self, t):
        self.logger.info('current optimizer iteration end ' + repr(t))
        for cb in self.tuning_callbacks:
            cb.on_iteration_end(t)

    @property
    def parameter_ranges(self):
        result = []
        for ranges in self._tunable_parameter_ranges.values():
            result.extend(ranges.values())
        return result

    def _set_new_parameters(self, point):
        i = 0
        # We are iterating over all elements of the dictionary in order.
        # This works because dicts are order-preserving.
        for obj, ranges in self._tunable_parameter_ranges.items():
            param_dict = dict()
            for j, p in enumerate(ranges.keys()):
                param_dict[p] = point[i + j]
            self.logger.info('obj: {}, current parameters {}'.format(type(obj).__name__, param_dict))
            obj.set_tunable_parameters(**param_dict)
            i += len(ranges)

    def _fit_ranker(self, xtrain, ytrain, xtest, ytest, next_point):
        start = datetime.now()
        self._set_new_parameters(next_point)
        try:
            self.learner.fit(xtrain, ytrain, **self._fit_params)
            ypred = self.learner(xtest)
            loss = get_mean_loss(self.validation_loss, ytest, ypred)
            time_taken = duration_till_now(start)
        except:
            self.logger.error(traceback.format_exc())
            self.logger.info("For current parameter error occurred so taking loss as maximum value")
            loss = 1.00
            time_taken = duration_till_now(start)
        return loss, time_taken

    def fit(self, X, Y, total_duration=600, n_iter=100, cv_iter=None, acq_func='gp_hedge', **kwargs):
        start = datetime.now()

        def splitter(itr):
            for train_idx, test_idx in itr:
                yield X[train_idx], Y[train_idx], X[test_idx], Y[test_idx]

        def splitter_dict(itr_dict):
            n_splits = len(list(itr_dict.values())[0])
            for i in range(n_splits):
                X_train = dict()
                Y_train = dict()
                X_test = dict()
                Y_test = dict()
                for n_obj, itr in itr_dict.items():
                    train_idx = itr[i][0]
                    test_idx = itr[i][1]
                    X_train[n_obj] = np.copy(X[n_obj][train_idx])
                    X_test[n_obj] = np.copy(X[n_obj][test_idx])
                    Y_train[n_obj] = np.copy(Y[n_obj][train_idx])
                    Y_test[n_obj] = np.copy(Y[n_obj][test_idx])
                yield X_train, Y_train, X_test, Y_test

        if cv_iter is None:
            cv_iter = ShuffleSplit(n_splits=3, test_size=0.1, random_state=self.random_state)
        if isinstance(X, dict):
            splits = dict()
            for n_obj, arr in X.items():
                if arr.shape[0] == 1:
                    splits[n_obj] = [([0], [0]) for i in range(cv_iter.n_splits)]
                else:
                    splits[n_obj] = list(cv_iter.split(arr))
        else:
            splits = list(cv_iter.split(X))
        # Pre-compute splits for reuse
        # Here we fix a random seed for all simulations to correlate the random
        # streams:

        seed = self.random_state.randint(2 ** 32, dtype='uint32')
        self.logger.debug('Random seed for the ranking algorithm: {}'.format(
            seed))
        opt_seed = self.random_state.randint(2 ** 32, dtype='uint32')
        self.logger.debug('Random seed for the optimizer: {}'.format(
            opt_seed
        ))
        gp_seed = self.random_state.randint(2 ** 32, dtype='uint32')
        self.logger.debug('Random seed for the GP surrogate: {}'.format(
            gp_seed
        ))
        n_iter = self.set_optimizer(n_iter, opt_seed, acq_func, gp_seed, **kwargs)

        self._callbacks_set_optimizer(self.opt)
        self._callbacks_on_optimization_begin()
        time_taken = duration_till_now(start)
        total_duration -= time_taken
        max_fit_duration = -np.inf
        self.logger.info('Time left for {} iterations is {}'.format(n_iter, seconds_to_time(total_duration)))

        try:
            for t in range(n_iter):
                if total_duration <= 0:
                    break
                start = datetime.now()
                self._callbacks_on_iteration_begin(t)
                self.logger.info('Starting optimization iteration: {}'.format(t))
                if t > 0:
                    self.log_best_params()
                next_point = self.opt.ask()
                self.logger.info('Next parameters:\n{}'.format(next_point))
                results = []
                running_times = []
                if isinstance(X, dict):
                    for X_train, Y_train, X_test, Y_test in splitter_dict(splits):
                        result, time_taken = self._fit_ranker(
                            X_train, Y_train,
                            X_test, Y_test, next_point)
                        running_times.append(time_taken)
                        results.append(result)
                else:
                    for X_train, Y_train, X_test, Y_test in splitter(splits):
                        result, time_taken = self._fit_ranker(
                            X_train, Y_train,
                            X_test, Y_test,
                            next_point)
                        running_times.append(time_taken)
                        results.append(result)

                results = np.array(results)
                running_times = np.array(running_times)
                mean_result = np.mean(results)
                mean_fitting_duration = np.mean(running_times)

                # Storing the maximum time to run the splitting model and adding the time for out of sample evaluation
                if max_fit_duration < np.sum(running_times):
                    max_fit_duration = np.sum(running_times)

                self.logger.info('Validation error for the parameters is {:.4f}'.format(mean_result))
                self.logger.info('Time taken for the parameters is {}'.format(seconds_to_time(np.sum(running_times))))
                if "ps" in self.opt.acq_func:
                    self.opt.tell(next_point, [mean_result, mean_fitting_duration])
                else:
                    self.opt.tell(next_point, mean_result)
                self._callbacks_on_iteration_end(t)

                self.logger.info(
                    "Main optimizer iterations done {} and saving the model".format(np.array(self.opt.yi).shape[0]))
                dump(self.opt, self.optimizer_path)

                time_taken = duration_till_now(start)
                total_duration -= time_taken
                self.logger.info('Time left for simulations is {} '.format(seconds_to_time(total_duration)))

                # Delete Tensorflow graph, to prevent memory leaks:
                K.clear_session()
                sess = tf.Session()
                K.set_session(sess)
                if (total_duration - max_fit_duration) < 0:
                    self.logger.info('Maximum time required by model to validate parameter values {}'.format(
                        seconds_to_time(max_fit_duration)))
                    self.logger.info('At iteration {} simulation stops, due to time deficiency'.format(t))
                    break

        except KeyboardInterrupt:
            self.logger.debug('Optimizer interrupted saving the model at {}'.format(self.optimizer_path))
            self.log_best_params()
        else:
            self.logger.debug('Finally, fit a model on the complete training set and storing the model at {}'.format(
                self.optimizer_path))

        finally:
            K.clear_session()
            sess = tf.Session()
            K.set_session(sess)
            self._callbacks_on_optimization_end()
            # self._fit_params["epochs"] = np.min([self._fit_params.get("epochs", 500) * 2, 1000])
            if "ps" in self.opt.acq_func:
                best_point = self.opt.Xi[np.argmin(np.array(self.opt.yi)[:, 0])]
            else:
                best_point = self.opt.Xi[np.argmin(self.opt.yi)]
            self._set_new_parameters(best_point)
            self.model = copy.copy(self.learner)
            self.model.fit(X, Y, **self._fit_params)
            if np.array(self.opt.yi).shape[0] != 0:
                dump(self.opt, self.optimizer_path)

    def log_best_params(self):
        if "ps" in self.opt.acq_func:
            best_i = np.argmin(np.array(self.opt.yi)[:, 0])
            best_loss = self.opt.yi[best_i]
            best_params = self.opt.Xi[best_i]
            self.logger.info("Best parameters so far with a loss of {:.4f}:\n {}".format(best_loss[0], best_params))
        else:
            best_i = np.argmin(self.opt.yi)
            best_loss = self.opt.yi[best_i]
            best_params = self.opt.Xi[best_i]
            self.logger.info("Best parameters so far with a loss of {:.4f}:\n {}".format(best_loss, best_params))

    def set_optimizer(self, n_iter, opt_seed, acq_func, gp_seed, **kwargs):
        self.logger.info('Retrieving model stored at: {}'.format(self.optimizer_path))
        try:
            optimizer = load(self.optimizer_path)
            self.logger.info('Loading model stored at: {}'.format(self.optimizer_path))
            finished_iter = np.array(optimizer.yi).shape[0]
            if finished_iter == 0:
                optimizer = None
                self.logger.info('Optimizer did not finish any iterations so setting optimizer to null')
        except KeyError:
            self.logger.error('Cannot open the file {}'.format(self.optimizer_path))
            optimizer = None

        except ValueError:
            self.logger.error('Cannot open the file {}'.format(self.optimizer_path))
            optimizer = None
        except FileNotFoundError:
            self.logger.error('No such file or directory: {}'.format(self.optimizer_path))
            optimizer = None

        if optimizer is not None:
            n_iter = n_iter - finished_iter
            if n_iter < 0:
                n_iter = 0
            self.logger.info('Iterations already done: {} and running iterations {}'.format(finished_iter, n_iter))
            self.opt = optimizer
            self.logger.debug('Setting the provided optimizer')
            self.log_best_params()
        else:
            transformed = []
            for param in self.parameter_ranges:
                transformed.append(check_dimension(param))
            self.logger.info("Parameter Space: {}".format(transformed))
            norm_space = normalize_dimensions(transformed)
            self.logger.info("Parameter Space after transformation: {}".format(norm_space))
            categorical_space = np.array([isinstance(s, Categorical) or isinstance(s, Integer) for s in norm_space])
            self.logger.info("categorical_space: {}".format(categorical_space))
            if np.all(categorical_space):
                base_estimator = cook_estimator("RF", space=norm_space, random_state=gp_seed)
            else:
                base_estimator = cook_estimator("GP", space=norm_space, random_state=gp_seed, noise="gaussian")

            self.opt = Optimizer(dimensions=self.parameter_ranges, random_state=opt_seed, base_estimator=base_estimator,
                                 acq_func=acq_func, **kwargs)

        return n_iter

    def predict_pair(self, a, b, **kwargs):
        if self.model is not None:
            return self.model.predict_pair(a, b, **kwargs)
        else:
            self.logger.error('The ranking model was not fit yet.')
            raise AttributeError

    def _predict_scores_fixed(self, X, **kwargs):
        if self.model is not None:
            return self.model._predict_scores_fixed(X, **kwargs)
        else:
            self.logger.error('The ranking model was not fit yet.')
            raise AttributeError

    def predict_scores(self, X, **kwargs):
        if self.model is not None:
            return self.model.predict_scores(X, **kwargs)
        else:
            self.logger.error('The ranking model was not fit yet.')
            raise AttributeError

    def predict(self, X, **kwargs):
        if self.model is not None:
            return self.model.predict(X, **kwargs)
        else:
            self.logger.error('The ranking model was not fit yet.')
            raise AttributeError

    def predict_for_scores(self, scores, **kwargs):
        if self.model is not None:
            return self.model.predict_for_scores(scores, **kwargs)
        else:
            self.logger.error('The ranking model was not fit yet.')
            raise AttributeError

    def set_tunable_parameters(self, **point):
        self.model.set_tunable_parameters(**point)


def check_learner_class(ranker):
    """ Function which checks if the ranker is an instance of the :class:`csrank.tunning.Tunable` class

    Parameters
    ----------
    ranker: object
        The ranker object to be checked
    """
    if not (isinstance(ranker, Tunable) and hasattr(ranker, 'set_tunable_parameters')):
        logging.error('The given object ranker is not tunable')
        raise AttributeError
