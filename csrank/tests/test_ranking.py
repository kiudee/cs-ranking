import os

from keras.optimizers import SGD
import numpy as np
import pytest
import tensorflow as tf

from csrank import PairedCombinatorialLogit
from csrank.constants import CMPNET
from csrank.constants import ERR
from csrank.constants import FATE_RANKER
from csrank.constants import FATELINEAR_RANKER
from csrank.constants import FETA_RANKER
from csrank.constants import FETALINEAR_RANKER
from csrank.constants import LISTNET
from csrank.constants import RANKNET
from csrank.constants import RANKSVM
from csrank.metrics_np import zero_one_accuracy_np
from csrank.metrics_np import zero_one_rank_loss_for_scores_ties_np
from csrank.objectranking import *
from csrank.objectranking.fate_object_ranker import FATEObjectRanker

optimizer = SGD(lr=1e-3, momentum=0.9, nesterov=True)

object_rankers = {
    FATELINEAR_RANKER: (
        FATELinearObjectRanker,
        {"n_hidden_set_units": 12, "batch_size": 1},
        (0.0, 1.0),
    ),
    FETALINEAR_RANKER: (FETALinearObjectRanker, {}, (0.0, 1.0)),
    FETA_RANKER: (
        FETAObjectRanker,
        {"add_zeroth_order_model": True, "optimizer": optimizer},
        (0.0, 1.0),
    ),
    RANKNET: (RankNet, {"optimizer": optimizer}, (0.0, 1.0)),
    CMPNET: (CmpNet, {"optimizer": optimizer}, (0.0, 1.0)),
    LISTNET: (ListNet, {"n_top": 3, "optimizer": optimizer}, (0.0, 1.0)),
    ERR: (ExpectedRankRegression, {}, (0.0, 1.0)),
    RANKSVM: (RankSVM, {}, (0.0, 1.0)),
    FATE_RANKER: (
        FATEObjectRanker,
        {
            "n_hidden_joint_layers": 1,
            "n_hidden_set_layers": 1,
            "n_hidden_joint_units": 5,
            "n_hidden_set_units": 5,
            "optimizer": optimizer,
        },
        (0.0, 1.0),
    ),
}


@pytest.fixture(scope="module")
def trivial_ranking_problem():
    random_state = np.random.RandomState(123)
    x = random_state.randn(2, 5, 1)
    y_true = x.argsort(axis=1).argsort(axis=1).squeeze(axis=-1)
    return x, y_true


def check_params_tunable(tunable_obj, params, rtol=1e-2, atol=1e-4):
    for key, value in params.items():
        if hasattr(tunable_obj, key):
            expected = getattr(tunable_obj, key)
            if isinstance(value, int) or isinstance(value, float):
                if (
                    isinstance(tunable_obj, PairedCombinatorialLogit)
                    and key == "n_nests"
                ):
                    tunable_obj.n_nests == tunable_obj.n_objects_fit_ * (
                        tunable_obj.n_objects_fit_ - 1
                    ) / 2
                else:
                    assert np.isclose(
                        expected, value, rtol=rtol, atol=atol, equal_nan=False
                    )
            else:
                assert value == expected
        elif key == "learning_rate" and hasattr(tunable_obj, "optimizer"):
            key = (
                "learning_rate"
                if "learning_rate" in tunable_obj.optimizer.get_config()
                else "lr"
            )
            learning_rate = tunable_obj.optimizer.get_config().get(key, 0.0)
            assert np.isclose(
                learning_rate, value, rtol=rtol, atol=atol, equal_nan=False
            )
        elif key == "reg_strength" and hasattr(tunable_obj, "kernel_regularizer"):
            config = tunable_obj.kernel_regularizer.get_config()
            val1 = np.isclose(
                config["l1"], value, rtol=rtol, atol=atol, equal_nan=False
            )
            val2 = np.isclose(
                config["l2"], value, rtol=rtol, atol=atol, equal_nan=False
            )
            assert val1 or val2


@pytest.mark.parametrize("ranker_name", list(object_rankers.keys()))
def test_object_ranker_fixed(trivial_ranking_problem, ranker_name):
    tf.set_random_seed(0)
    os.environ["KERAS_BACKEND"] = "tensorflow"
    np.random.seed(123)
    x, y = trivial_ranking_problem
    ranker, params, (loss, acc) = object_rankers[ranker_name]
    ranker = ranker(**params)
    if "linear" in ranker_name:
        ranker.fit(x, y, epochs=10, validation_split=0, verbose=False)
    else:
        ranker.fit(x, y, epochs=100, validation_split=0, verbose=False)
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
    params = {
        "n_hidden": 20,
        "n_units": 20,
        "n_hidden_set_units": 2,
        "n_hidden_set_layers": 10,
        "n_hidden_joint_units": 2,
        "n_hidden_joint_layers": 10,
        "reg_strength": 1e-3,
        "learning_rate": 1e-1,
        "batch_size": 32,
        "alpha": 0.5,
        "l1_ratio": 0.7,
        "tol": 1e-2,
        "C": 10,
        "n_mixtures": 10,
        "n_nests": 5,
        "regularization": "l2",
    }
    ranker.set_tunable_parameters(**params)
    check_params_tunable(ranker, params, rtol, atol)
