import numpy as np
import pytest
import torch
from torch import optim

from csrank.constants import CMPNET
from csrank.constants import ERR
from csrank.constants import FATE_RANKER
from csrank.constants import FETA_RANKER
from csrank.constants import RANKNET
from csrank.constants import RANKSVM
from csrank.metrics_np import zero_one_accuracy_np
from csrank.metrics_np import zero_one_rank_loss_for_scores_ties_np
from csrank.objectranking import CmpNetObjectRanker
from csrank.objectranking import ExpectedRankRegression
from csrank.objectranking import FATEObjectRanker
from csrank.objectranking import FETAObjectRanker
from csrank.objectranking import RankNetObjectRanker
from csrank.objectranking import RankSVM

skorch_common_args = {
    "max_epochs": 100,
    "optimizer": optim.SGD,
    "optimizer__lr": 1e-3,
    "optimizer__momentum": 0.9,
    "optimizer__nesterov": True,
    # We evaluate the estimators in-sample. These tests are just small
    # sanity checks, so overfitting is okay here.
    "train_split": None,
}

object_rankers = {
    CMPNET: (
        CmpNetObjectRanker,
        {"n_hidden": 2, "n_units": 8, **skorch_common_args},
        (0.0, 1.0),
    ),
    ERR: (ExpectedRankRegression, {}, (0.0, 1.0)),
    RANKSVM: (RankSVM, {}, (0.0, 1.0)),
    FATE_RANKER: (
        FATEObjectRanker,
        {
            "n_hidden_joint_layers": 1,
            "n_hidden_set_layers": 1,
            "n_hidden_joint_units": 5,
            "n_hidden_set_units": 5,
            **skorch_common_args,
        },
        (0.0, 1.0),
    ),
    FETA_RANKER: (
        FETAObjectRanker,
        {
            "n_hidden": 1,
            "n_units": 8,
            "add_zeroth_order_model": False,
            **skorch_common_args,
        },
        (0.0, 1.0),
    ),
    FETA_RANKER
    + "zeroth_order_model": (
        FETAObjectRanker,
        {
            "n_hidden": 1,
            "n_units": 8,
            "add_zeroth_order_model": True,
            **skorch_common_args,
        },
        (0.0, 1.0),
    ),
    RANKNET: (
        RankNetObjectRanker,
        {"n_hidden": 2, "n_units": 8, **skorch_common_args},
        (0.0, 1.0),
    ),
}


@pytest.fixture(scope="module")
def trivial_ranking_problem():
    random_state = np.random.RandomState(123)
    # pytorch uses 32 bit floats by default. That should be precise enough and
    # makes it easier to use pytorch and non-pytorch estimators interchangeably.
    x = random_state.randn(2, 5, 1).astype(np.float32)
    y_true = x.argsort(axis=1).argsort(axis=1).squeeze(axis=-1)
    return x, y_true


@pytest.mark.parametrize("ranker_name", list(object_rankers.keys()))
def test_object_ranker_fixed(trivial_ranking_problem, ranker_name):
    np.random.seed(123)
    # There are some caveats with pytorch reproducibility. See the comment on
    # the corresponding line of `test_choice_functions.py` for details.
    torch.manual_seed(123)
    torch.use_deterministic_algorithms(True)
    x, y = trivial_ranking_problem
    ranker, params, (loss, acc) = object_rankers[ranker_name]
    ranker = ranker(**params)
    ranker.fit(x, y)
    pred_scores = ranker.predict_scores(x)
    pred_loss = zero_one_rank_loss_for_scores_ties_np(y, pred_scores)
    rtol = 1e-2
    atol = 1e-4
    assert np.isclose(loss, pred_loss, rtol=rtol, atol=atol, equal_nan=False)
    pred = ranker.predict_for_scores(pred_scores)
    pred_2 = ranker.predict(x)
    pred_acc = zero_one_accuracy_np(pred, pred_2)
    assert np.isclose(1.0, pred_acc, rtol=rtol, atol=atol, equal_nan=False)
    pred_acc = zero_one_accuracy_np(pred, y)
    assert np.isclose(acc, pred_acc, rtol=rtol, atol=atol, equal_nan=False)
