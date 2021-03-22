from .baseline import AllPositive
from .cmpnet_choice import CmpNetChoiceFunction
from .fate_choice import FATEChoiceFunction
from .fatelinear_choice import FATELinearChoiceFunction
from .feta_choice import FETAChoiceFunction
from .fetalinear_choice import FETALinearChoiceFunction
from .generalized_linear_model import GeneralizedLinearModel
from .pairwise_choice import PairwiseSVMChoiceFunction
from .ranknet_choice import RankNetChoiceFunction

algorithms = [
    AllPositive,
    CmpNetChoiceFunction,
    FATEChoiceFunction,
    FATELinearChoiceFunction,
    FETAChoiceFunction,
    FETALinearChoiceFunction,
    GeneralizedLinearModel,
    PairwiseSVMChoiceFunction,
    RankNetChoiceFunction,
]

__all__ = [algo.__name__ for algo in algorithms]
