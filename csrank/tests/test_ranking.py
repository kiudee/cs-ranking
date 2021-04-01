import numpy as np
import pytest
import torch

from csrank.constants import ERR
from csrank.constants import RANKSVM
from csrank.metrics_np import zero_one_accuracy_np
from csrank.metrics_np import zero_one_rank_loss_for_scores_ties_np
from csrank.objectranking import *

object_rankers = {
    ERR: (ExpectedRankRegression, {}, (0.0, 1.0)),
    RANKSVM: (RankSVM, {}, (0.0, 1.0)),
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
