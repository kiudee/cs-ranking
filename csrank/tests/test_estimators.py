"""Check that our estimators adhere to the scikit-learn interface.

https://scikit-learn.org/stable/developers/develop.html#rolling-your-own-estimator
"""

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
            return super().fit(Xnew, Ynew, *args, **kwargs)

        def predict(self, X, *args, **kwargs):
            super().predict(_reshape_x(X), *args, **kwargs)

    for (estimator, check) in check_estimator(WrappedRanker, generate_only=True):
        # checks that attempt to call "fit" do not work since our estimators
        # expect a 3-dimensional data shape while scikit-learn assumes two
        # dimensions (an array of 1d data).
        if not get_check_name(check) in {
            "check_estimators_fit_returns_self",  # fails for all
            "check_complex_data",  # fails for CmpNet
            "check_dtype_object",  # fails for ExpectedRankRegression
            "check_estimators_empty_data_messages",  # fails for all
            "check_estimators_nan_inf",  # fails for CmpNet
            "check_estimators_overwrite_params",  # fails for FATELinearObjectRanker
            "check_estimator_sparse_data",  # fails for ExpectedRankRegression
            "check_estimators_pickle",  # fails for ExpectedRankRegression
            "check_fit2d_predict1d",  # fails for ExpectedRankRegression
            "check_methods_subset_invariance",  # fails for ExpectedRankRegression
            "check_fit2d_1sample",  # fails for FETAObjectRanker
            "check_dict_unchanged",  # fails for ListNet
            "check_dont_overwrite_parameters",  # fails for CmpNet
            "check_fit_idempotent",  # fails for ExpectedRankRegression
            "check_n_features_in" # fails for RankSVM
        }:
            check(estimator)
