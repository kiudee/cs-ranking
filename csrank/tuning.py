import logging
from datetime import datetime

import numpy as np
from keras.losses import categorical_hinge
from sklearn.model_selection import ShuffleSplit
from sklearn.utils import check_random_state
from skopt import Optimizer
from skopt.space import check_dimension
from skopt.utils import cook_estimator, normalize_dimensions, dump

from csrank.constants import OBJECT_RANKING, LABEL_RANKING, DISCRETE_CHOICE, DYAD_RANKING
from csrank.fate_ranking import FATEObjectRanker, FATEContextualRanker, FATEObjectChooser, \
    FATELabelRanker
from csrank.metrics import zero_one_rank_loss
from csrank.objectranking.object_ranker import ObjectRanker
from csrank.tunable import check_ranker_class
from csrank.util import duration_tillnow, create_dir_recursively, microsec_to_time, \
    get_mean_loss_for_dictionary, get_loss_for_array

PARAMETER_OPTIMIZER = "ParameterOptimizer"


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


def fit_parallel(xtrain, ytrain, xtest, ytest, seed, next_point,
                 ranker_class, ranker_params, fit_params, validation_loss):
    start = datetime.now()
    ranker = ranker_class(random_state=seed, **ranker_params)
    ranker.set_tunable_parameters(next_point)
    ranker.fit(xtrain, ytrain, **fit_params)
    ypred = ranker(xtest)
    if isinstance(xtest, dict):
        loss = get_mean_loss_for_dictionary(logging.getLogger(PARAMETER_OPTIMIZER), validation_loss, ytest, ypred)
    else:
        loss = get_loss_for_array(validation_loss, ytest, ypred)
    time_taken = duration_tillnow(start)
    return loss, time_taken


class ParameterOptimizer(ObjectRanker):
    def __init__(self, ranker_class, optimizer_path, ranker_params=None, fit_params=None,
                 random_state=None, tuning_callbacks=None, validation_loss=None, learning_problem=OBJECT_RANKING,
                 **kwd):
        self.logger = logging.getLogger(PARAMETER_OPTIMIZER)

        default_rankers = {OBJECT_RANKING: FATEObjectRanker, LABEL_RANKING: FATELabelRanker,
                           DISCRETE_CHOICE: FATEObjectChooser, DYAD_RANKING: FATEContextualRanker}
        create_dir_recursively(optimizer_path, True)
        self.optimizer_path = optimizer_path

        if ranker_class is None:
            self._ranker_class = default_rankers[learning_problem]
        else:
            check_ranker_class(ranker_class)
            self._ranker_class = ranker_class

        if ranker_params is None:
            raise ValueError('Ranker parameters cannot be Empty')
        else:
            self._ranker_params = ranker_params

        if tuning_callbacks is None:
            self.tuning_callbacks = []
        else:
            self.tuning_callbacks = tuning_callbacks
        default_validation_loss = {OBJECT_RANKING: zero_one_rank_loss, LABEL_RANKING: zero_one_rank_loss,
                                   DISCRETE_CHOICE: categorical_hinge, DYAD_RANKING: zero_one_rank_loss}
        if validation_loss is None:
            self.validation_loss = default_validation_loss[learning_problem]
            self.logger.info(
                'Loss function is not specified, using {}'.format(default_validation_loss[learning_problem].__name__))
        else:
            self.validation_loss = validation_loss

        if fit_params is None:
            self._fit_params = {}
            self.logger.warning("Fit params are empty, the default parameters will be applied")
        else:
            self._fit_params = fit_params

        self.random_state = check_random_state(random_state)
        self.model = None

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
        self.logger.info('********************** optimizer iteration begin ********************** ' + repr(t))
        for cb in self.tuning_callbacks:
            cb.on_iteration_begin(t)

    def _callbacks_on_iteration_end(self, t):
        self.logger.info('current optimizer iteration end ' + repr(t))
        for cb in self.tuning_callbacks:
            cb.on_iteration_end(t)

    def fit(self, X, Y, total_duration, n_iter=100, cv_iter=None, optimizer=None, acq_func='gp_hedge',
            parameters_ranges=dict(),
            **kwargs):
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
            cv_iter = ShuffleSplit(n_splits=3, test_size=0.1,
                                   random_state=self.random_state)
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
        if "use_early_stopping" in self._ranker_params:
            self._ranker_class._use_early_stopping = self._ranker_params["use_early_stopping"]
        param_ranges = self._ranker_class.set_tunable_parameter_ranges(parameters_ranges)

        if (optimizer is not None):
            opt = optimizer
            self.logger.debug('Setting the provided optimizer')
            self.log_best_params(opt)
        else:
            transformed = []
            for param in param_ranges:
                transformed.append(check_dimension(param))
            self.logger.info("Parameter Space: {}".format(transformed))
            space = normalize_dimensions(transformed)
            self.logger.info("Parameter Space after transformation: {}".format(space))

            # Todo: Make this passable
            base_estimator = cook_estimator("GP", space=space, random_state=gp_seed, noise="gaussian")
            opt = Optimizer(dimensions=param_ranges, random_state=opt_seed,
                            base_estimator=base_estimator, acq_func=acq_func, **kwargs)
        self._callbacks_set_optimizer(opt)
        self._callbacks_on_optimization_begin()
        time_taken = duration_tillnow(start)
        total_duration -= time_taken
        max_fit_duration = -10000
        self.logger.info('Time left for {} iterations is {}'.format(n_iter, microsec_to_time(total_duration)))

        try:
            for t in range(n_iter):
                start = datetime.now()
                self._callbacks_on_iteration_begin(t)
                self.logger.info(
                    'Starting optimization iteration: {}'.format(t))
                if t > 0:
                    self.log_best_params(opt)

                next_point = opt.ask()
                self.logger.info('Next parameters:\n{}'.format(next_point))
                results = []
                running_times = []
                if isinstance(X, dict):
                    for X_train, Y_train, X_test, Y_test in splitter_dict(splits):
                        result, time_taken = fit_parallel(X_train, Y_train, X_test, Y_test, seed, next_point,
                                                          self._ranker_class,
                                                          self._ranker_params, self._fit_params, self.validation_loss)
                        running_times.append(time_taken)
                        results.append(result)
                else:
                    for X_train, Y_train, X_test, Y_test in splitter(splits):
                        result, time_taken = fit_parallel(X_train, Y_train, X_test, Y_test, seed, next_point,
                                                          self._ranker_class,
                                                          self._ranker_params, self._fit_params, self.validation_loss)
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
                self.logger.info('Time taken for the parameters is {}'.format(microsec_to_time(np.sum(running_times))))
                if "ps" in opt.acq_func:
                    opt.tell(next_point, [mean_result, mean_fitting_duration])
                else:
                    opt.tell(next_point, mean_result)
                self._callbacks_on_iteration_end(t)

                self.logger.info(
                    "Main optimizer iterations done {} and saving the model".format(np.array(opt.yi).shape[0]))
                dump(opt, self.optimizer_path)

                time_taken = duration_tillnow(start)
                total_duration -= time_taken
                self.logger.info('Time left for simulations is {} '.format(microsec_to_time(total_duration)))

                if (total_duration - max_fit_duration) < 0:
                    self.logger.info(
                        'At iteration {} maximum time required by model to validate a parameter values'.format(
                            microsec_to_time(max_fit_duration)))
                    self.logger.info('At iteration {} simulation stops, due to time deficiency'.format(t))
                    break

        except KeyboardInterrupt:
            self.logger.debug('Optimizer interrupted saving the model at {}'.format(self.optimizer_path))
            self.log_best_params(opt)
            self.optimizer = opt
            if np.array(opt.yi).shape[0] != 0:
                dump(opt, self.optimizer_path)

        else:
            self.logger.debug('Finally, fit a model on the complete training set and storing the model at {}'.format(
                self.optimizer_path))
            self._fit_params["epochs"] = self._fit_params.get("epochs", 1000) * 2
            self.model = self._ranker_class(random_state=self.random_state, **self._ranker_params)
            if "ps" in opt.acq_func:
                best_point = opt.Xi[np.argmin(np.array(opt.yi)[:, 0])]
            else:
                best_point = opt.Xi[np.argmin(opt.yi)]
            self.model.set_tunable_parameters(best_point)
            self.model.fit(X, Y, **self._fit_params)
            self.optimizer = opt
            if np.array(opt.yi).shape[0] != 0:
                dump(opt, self.optimizer_path)

        finally:
            self._callbacks_on_optimization_end()
            self.optimizer = opt
            if np.array(opt.yi).shape[0] != 0:
                dump(opt, self.optimizer_path)

    def log_best_params(self, opt):
        if "ps" in opt.acq_func:
            best_i = np.argmin(np.array(opt.yi)[:, 0])
            best_loss = opt.yi[best_i]
            best_params = opt.Xi[best_i]
            self.logger.info(
                "Best parameters so far with a loss of {:.4f} time of {:.4f}:\n {}".format(best_loss[0],
                                                                                           best_loss[1],
                                                                                           best_params))
        else:
            best_i = np.argmin(opt.yi)
            best_loss = opt.yi[best_i]
            best_params = opt.Xi[best_i]
            self.logger.info(
                "Best parameters so far with a loss of {:.4f}:\n {}".format(best_loss, best_params))

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
