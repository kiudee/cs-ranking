import numpy as np
from pymc3.variational.callbacks import CheckParametersConvergence
import pytest
import torch

from csrank.choicefunction import *
from csrank.constants import GLM_CHOICE
from csrank.constants import RANKSVM_CHOICE
from csrank.metrics_np import auc_score
from csrank.metrics_np import f1_measure
from csrank.metrics_np import instance_informedness
from csrank.metrics_np import subset_01_loss
from csrank.util import metrics_on_predictions

choice_metrics = {
    "F1Score": f1_measure,
    "Informedness": instance_informedness,
    "AucScore": auc_score,
}


def get_vals(values):
    return dict(zip(choice_metrics.keys(), values))


choice_functions = {
    GLM_CHOICE: (GeneralizedLinearModel, {}, get_vals([0.9567, 0.9955, 1.0])),
    RANKSVM_CHOICE: (PairwiseSVMChoiceFunction, {}, get_vals([0.9522, 0.9955, 1.0])),
}


@pytest.fixture(scope="module")
def trivial_choice_problem():
    random_state = np.random.RandomState(42)
    # pytorch uses 32 bit floats by default. That should be precise enough and
    # makes it easier to use pytorch and non-pytorch estimators interchangeably.
    x = random_state.randn(200, 5, 1).astype(np.float32)
    # The pytorch estimators expect booleans to be encoded as a 32 bit float
    # (1.0 for True, 0.0 for false).
    y_true = np.array(x.squeeze(axis=-1) > np.mean(x), dtype=np.float32)
    return x, y_true


@pytest.mark.parametrize("name", list(choice_functions.keys()))
def test_choice_function_fixed(trivial_choice_problem, name):
    np.random.seed(123)
    # Pytorch does not guarantee full reproducibility in different settings
    # [1]. This may become a problem in the test suite, in which case we should
    # increase the tolerance. These are only "sanity checks" on small data sets
    # anyway and the exact values do not mean much here.
    # [1] https://pytorch.org/docs/stable/notes/randomness.html
    torch.manual_seed(123)
    # Trade off performance for better reproducibility.
    torch.use_deterministic_algorithms(True)
    x, y = trivial_choice_problem
    choice_function = choice_functions[name][0]
    params, accuracies = choice_functions[name][1], choice_functions[name][2]
    learner = choice_function(**params)
    if name == GLM_CHOICE:
        learner.fit(
            x,
            y,
            vi_params={
                "n": 100,
                "method": "advi",
                "callbacks": [CheckParametersConvergence()],
            },
        )
    else:
        learner.fit(x, y)

    s_pred = learner.predict_scores(x)
    y_pred = learner.predict_for_scores(s_pred)
    y_pred_2 = learner.predict(x)
    rtol = 1e-2
    atol = 5e-2
    assert np.isclose(
        0.0, subset_01_loss(y_pred, y_pred_2), rtol=rtol, atol=atol, equal_nan=False
    )
    for key, value in accuracies.items():
        metric = choice_metrics[key]
        if metric in metrics_on_predictions:
            pred_loss = metric(y, y_pred)
        else:
            pred_loss = metric(y, s_pred)
        assert np.isclose(value, pred_loss, rtol=rtol, atol=atol, equal_nan=False)
