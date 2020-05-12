"""Check that our estimators adhere to the scikit-learn interface.

https://scikit-learn.org/stable/developers/develop.html#rolling-your-own-estimator
"""

import pytest
from sklearn.utils.estimator_checks import check_parameters_default_constructible

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
    check_parameters_default_constructible("default_constructible", Estimator)
