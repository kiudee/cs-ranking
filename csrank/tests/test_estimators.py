"""Check that our estimators adhere to the scikit-learn interface.

https://scikit-learn.org/stable/developers/develop.html#rolling-your-own-estimator
"""

from functools import partial

import pytest
from sklearn.utils.estimator_checks import check_estimator

from csrank.choicefunction import CmpNetChoiceFunction
from csrank.choicefunction import FATEChoiceFunction
from csrank.choicefunction import FATELinearChoiceFunction
from csrank.choicefunction import FETAChoiceFunction
from csrank.choicefunction import FETALinearChoiceFunction
from csrank.choicefunction import GeneralizedLinearModel
from csrank.choicefunction import PairwiseSVMChoiceFunction
from csrank.choicefunction import RankNetChoiceFunction
from csrank.discretechoice import CmpNetDiscreteChoiceFunction
from csrank.discretechoice import FATEDiscreteChoiceFunction
from csrank.discretechoice import FATELinearDiscreteChoiceFunction
from csrank.discretechoice import FETADiscreteChoiceFunction
from csrank.discretechoice import FETALinearDiscreteChoiceFunction
from csrank.discretechoice import GeneralizedNestedLogitModel
from csrank.discretechoice import MixedLogitModel
from csrank.discretechoice import MultinomialLogitModel
from csrank.discretechoice import NestedLogitModel
from csrank.discretechoice import PairedCombinatorialLogit
from csrank.discretechoice import PairwiseSVMDiscreteChoiceFunction
from csrank.discretechoice import RankNetDiscreteChoiceFunction
from csrank.objectranking import CmpNet
from csrank.objectranking import ExpectedRankRegression
from csrank.objectranking import FATELinearObjectRanker
from csrank.objectranking import FATEObjectRanker
from csrank.objectranking import FETALinearObjectRanker
from csrank.objectranking import FETAObjectRanker
from csrank.objectranking import ListNet
from csrank.objectranking import RankNet
from csrank.objectranking import RankSVM


@pytest.mark.parametrize(
    "Estimator",
    [
        CmpNet,
        CmpNetChoiceFunction,
        CmpNetDiscreteChoiceFunction,
        ExpectedRankRegression,
        FATEChoiceFunction,
        FATEDiscreteChoiceFunction,
        FATELinearChoiceFunction,
        FATELinearDiscreteChoiceFunction,
        FATELinearObjectRanker,
        FATEObjectRanker,
        FETAChoiceFunction,
        FETADiscreteChoiceFunction,
        FETALinearChoiceFunction,
        FETALinearDiscreteChoiceFunction,
        FETALinearObjectRanker,
        FETAObjectRanker,
        GeneralizedLinearModel,
        GeneralizedNestedLogitModel,
        ListNet,
        MixedLogitModel,
        MultinomialLogitModel,
        NestedLogitModel,
        PairedCombinatorialLogit,
        PairwiseSVMChoiceFunction,
        PairwiseSVMDiscreteChoiceFunction,
        RankNet,
        RankNetChoiceFunction,
        RankNetDiscreteChoiceFunction,
        RankSVM,
    ],
)
def test_all_estimators(Estimator):
    def get_check_name(check):
        if isinstance(check, partial):
            return check.func.__name__
        else:
            return check.__name__

    for (estimator, check) in check_estimator(Estimator, generate_only=True):
        # checks that attempt to call "fit" do not work since our estimators
        # expect a 3-dimensional data shape while scikit-learn assumes two
        # dimensions (an array of 1d data).
        if not get_check_name(check) in {
            "check_estimators_dtypes",
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
