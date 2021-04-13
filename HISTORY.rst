=======
History
=======

2.0.0 (Unreleased)
------------------

* The library has been migrated to pytorch. This is a breaking change. You will
  likely need to adapt to this new version if you have been using estimators
  from version 1.x.

* The RankNet and CmpNet estimators are now trained with a loss that applies to
  the whole result (the general/discrete choice or ranking). They were
  previously trained on object pairs with different loss functions.

* Behavior and default parameters of the estimators may differ from the
  previous versions. For example the default activation for CmpNet and RankNet
  is now SELU instead of ReLU.

1.3.0 (Unreleased)
------------------

* We no longer override any of the defaults of our default optimizer (SGD). In
  particular, the parameters nesterov, momentum and lr are now set to the
  default values set by keras.

* All optimizers must now be passed in uninitialized. Optimizer parameters can
  be set by passing `optimizer__{kwarg}` parameters to the learner. This
  follows the scikit-learn and skorch standard.

* Regularizers must similarly be passed uninitialized, therefore the
  `reg_strength` parameter is replaced by `kernel_regularizer__l`.

* Tuning functionality has been removed. Since our Learners are ScikitLearn
  estimators, any standard tuning framework should work and no special support
  is needed.

* The tunable class and notably its `set_tunable_parameters` function has been
  removed. Use `set_params` from the scikit-learn estimator API instead.

1.2.1 (2020-06-08)
------------------

* Make all our optional dependencies mandatory to work around a bug in our
  optional imports code. Without this, an exception is raised on import.
  A proper fix will follow.

1.2.0 (2020-06-05)
------------------

* Change public interface of the learners to be more in line with the
  scikit-learn interface (ongoing). As part of these changes, it is no longer
  required to explicitly pass the data dimensionality to the learners on
  initialization.
* Rewrite and document normalized discounted cumulative gain (ndcg) metric to
  fix numerical issues.
  See `#32 <https://github.com/kiudee/cs-ranking/issues/32>`__ for details.
* Fix passing fit keyword arguments on to the core network in
  ``FATEChoiceFunction``.
* Fix arguments for ``AllPositive`` baseline.
* Raise ValueError rather than silently using a default value for unknown
  passed arguments.
* Internal efforts to increase code quality and make use of linting
  (``black``, ``flake8``, ``doc8``).
* Remove old experimental code.

1.1.0 (2020-03-19)
------------------

* Add the expected reciprocal rank (ERR) metric.
* Fix bug in callbacks causing the wrong learning rate schedule to be applied.
* Make csrank easier to install by making some dependencies optional.
* Add guidelines for how to contribute to the project.

1.0.2 (2020-02-12)
------------------

* Fix deployment to GH-pages

1.0.1 (2020-02-03)
------------------

* Add ``HISTORY.rst`` file to track changes over time
* Set up travis-ci for deployment to PyPi

1.0.0 (2018-03-05)
------------------

* Initial release
