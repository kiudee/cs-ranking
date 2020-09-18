"""Check that our estimators adhere to the scikit-learn interface.

https://scikit-learn.org/stable/developers/develop.html#rolling-your-own-estimator
"""

import numpy as np

from functools import partial

import pytest
from sklearn.utils.estimator_checks import check_estimator

from csrank import objectranking


def get_check_name(check):
    if isinstance(check, partial):
        return check.func.__name__
    else:
        return check.__name__


def _reshape_x(X):
    n_instances, n_objects = X.shape
    n_features = 1
    return X.reshape((n_instances, n_objects, n_features))


@pytest.mark.parametrize(
    "Estimator",
    # TODO write wrappers for choice, discretechoice
    objectranking.algorithms,
)
def test_all_estimators(Estimator):
    class WrappedRanker(Estimator):
        # scikit learn assumes that "X" is an array of one-dimensional
        # feature vectors. Our learners however assume an array of objects
        # as a "feature vector", hence they expect one more dimension.
        # This is one scikit-learn API expectation that we do not fulfill.
        # This thin wrapper is needed so that we can still use the other
        # estimator checks. It just pretends every feature is itself a
        # one-feature object.
        def fit(self, X, Y, *args, **kwargs):
            Xnew = _reshape_x(X)
            Ynew = Xnew.argsort(axis=1).argsort(axis=1).squeeze(axis=-1)
            super().fit(Xnew, Ynew, *args, **kwargs)

        def predict(self, X, *args, **kwargs):
            super().predict(_reshape_x(X), *args, **kwargs)

    for (estimator, check) in check_estimator(WrappedRanker, generate_only=True):
        # checks that attempt to call "fit" do not work since our estimators
        # expect a 3-dimensional data shape while scikit-learn assumes two
        # dimensions (an array of 1d data).
        if not get_check_name(check) in {
            # "check_estimators_dtypes",
            "check_fit_score_takes_y",
            "check_estimators_fit_returns_self",
            "check_complex_data",
            "check_dtype_object",
            "check_estimators_empty_data_messages",
            "check_pipeline_consistency",
            "check_estimators_nan_inf",
            "check_estimators_overwrite_params",
            "check_estimator_sparse_data",
            "check_estimators_pickle",
            "check_fit2d_predict1d",
            "check_methods_subset_invariance",
            "check_fit2d_1sample",
            "check_fit2d_1feature",
            "check_dict_unchanged",
            "check_dont_overwrite_parameters",
            "check_fit_idempotent",
            "check_fit1d",
        }:
            check(estimator)
