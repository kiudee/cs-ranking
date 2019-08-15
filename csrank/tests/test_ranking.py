import os

import numpy as np
import pytest
import tensorflow as tf
from keras.optimizers import SGD

from csrank import PairedCombinatorialLogit
from csrank.experiments.constants import *
from csrank.metrics_np import zero_one_rank_loss_for_scores_ties_np, zero_one_accuracy_np
from csrank.objectranking import *
from csrank.objectranking.fate_object_ranker import FATEObjectRanker

optimizer = SGD(lr=1e-3, momentum=0.9, nesterov=True)

object_rankers = {
    FATELINEAR_RANKER: (FATELinearObjectRanker, {"n_hidden_set_units": 128, "batch_size": 32}, (1.0, 0.0)),
    FETALINEAR_RANKER: (FETALinearObjectRanker, {}, (0.9112, 0.0)),
    FETA_RANKER: (FETAObjectRanker, {"add_zeroth_order_model": True, "optimizer": optimizer}, (0.0, 1.0)),
    RANKNET: (RankNet, {"optimizer": optimizer}, (0.0, 1.0)),
    CMPNET: (CmpNet, {"optimizer": optimizer}, (0.0, 1.0)),
    LISTNET: (ListNet, {"n_top": 3, "optimizer": optimizer}, (0.0, 1.0)),
    ERR: (ExpectedRankRegression, {}, (0.0, 1.0)),
    RANKSVM: (RankSVM, {}, (0.0, 1.0)),
    FATE_RANKER: (FATEObjectRanker, {"n_hidden_joint_layers": 1, "n_hidden_set_layers": 1, "n_hidden_joint_units": 5,
                                     "n_hidden_set_units": 5, "optimizer": optimizer}, (0.0, 1.0))
}


@pytest.fixture(scope="module")
def trivial_ranking_problem():
    random_state = np.random.RandomState(123)
    x = random_state.randn(200, 5, 1)
    y_true = x.argsort(axis=1).argsort(axis=1).squeeze(axis=-1)
    return x, y_true


def check_leaner(ranker, params, rtol=1e-2, atol=1e-4):
    for key, value in params.items():
        if key in ranker.__dict__.keys():
            expected = ranker.__dict__[key]
            if isinstance(value, int) or isinstance(value, float):
                if isinstance(ranker, PairedCombinatorialLogit) and key == "n_nests":
                    ranker.n_nests == ranker.n_objects * (ranker.n_objects - 1) / 2
                else:
                    assert np.isclose(expected, value, rtol=rtol, atol=atol, equal_nan=False)
            else:
                assert value == expected
        elif key == "learning_rate" and "optimizer" in ranker.__dict__.keys():
            assert np.isclose(ranker.optimizer.get_config()['lr'], value, rtol=rtol, atol=atol, equal_nan=False)
        elif key == "reg_strength" and "kernel_regularizer" in ranker.__dict__.keys():
            config = ranker.kernel_regularizer.get_config()
            val1 = np.isclose(config["l1"], value, rtol=rtol, atol=atol, equal_nan=False)
            val2 = np.isclose(config["l2"], value, rtol=rtol, atol=atol, equal_nan=False)
            assert val1 or val2


@pytest.mark.parametrize(
    "ranker_name", list(object_rankers.keys())
)
def test_object_ranker_fixed(trivial_ranking_problem, ranker_name):
    tf.set_random_seed(0)
    os.environ["KERAS_BACKEND"] = "tensorflow"
    np.random.seed(123)
    x, y = trivial_ranking_problem
    ranker, params, (loss, acc) = object_rankers[ranker_name]
    params["n_objects"], params["n_object_features"] = tuple(x.shape[1:])
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
    params = {"n_hidden": 20, "n_units": 20, "n_hidden_set_units": 2, "n_hidden_set_layers": 10,
              "n_hidden_joint_units": 2, "n_hidden_joint_layers": 10, "reg_strength": 1e-3, "learning_rate": 1e-1,
              "batch_size": 32, "alpha": 0.5, "l1_ratio": 0.7, "tol": 1e-2, "C": 10, "n_mixtures": 10, "n_nests": 5,
              "regularization": "l2"}
    ranker.set_tunable_parameters(**params)
    check_leaner(ranker, params, rtol, atol)
