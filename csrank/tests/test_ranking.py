import os
from abc import ABCMeta

import numpy as np
import pytest
import tensorflow as tf
from keras.layers import Input
from keras.models import Model
from keras.optimizers import SGD
from keras.regularizers import l2

from csrank import (
    FETANetwork, RankNet, CmpNet, ExpectedRankRegression, RankSVM, ListNet
)
from csrank.metrics_np import zero_one_rank_loss_for_scores_ties_np
from csrank.objectranking.fate_object_ranker import FATEObjectRanker
from ..fate_network import FATENetworkCore

RANKSVM = "ranksvm"
ERR = "err"
CMPNET = "cmpnet"
RANKNET = "ranknet"
LISTNET = "listnet"
FETA_RANKER = "feta_ranker"
FATE_RANKER = "fate_ranker"

object_rankers = {
    FETA_RANKER: FETANetwork,
    RANKNET: RankNet,
    CMPNET: CmpNet,
    LISTNET: ListNet,
    ERR: ExpectedRankRegression,
    RANKSVM: RankSVM,
    FATE_RANKER: FATEObjectRanker,
}
optimizer = SGD(lr=1e-3, momentum=0.9, nesterov=True)
object_rankers_params = {
    FETA_RANKER: {"add_zeroth_order_model": True, "optimizer": optimizer},
    RANKNET: {"optimizer": optimizer},
    CMPNET: {"optimizer": optimizer},
    FATE_RANKER: {
        "n_hidden_joint_layers": 1,
        "n_hidden_set_layers": 1,
        "n_hidden_joint_units": 5,
        "n_hidden_set_units": 5,
        "optimizer": optimizer,
    },
    ERR: {},
    RANKSVM: {},
    LISTNET: {"n_top": 3}
}


def test_construction_core():
    n_objects = 3
    n_features = 2

    # Create mock class:

    class MockClass(FATENetworkCore, metaclass=ABCMeta):

        def set_tunable_parameters(self, point):
            super().set_tunable_parameters(point)

        def predict(self, *args, **kwargs):
            pass

        def fit(self, *args, **kwargs):
            pass

    grc = MockClass(n_objects=n_objects, n_features=n_features)
    grc._construct_layers(
        activation=grc.activation,
                          kernel_initializer=grc.kernel_initializer,
        kernel_regularizer=grc.kernel_regularizer,
    )
    input_layer = Input(shape=(n_objects, n_features))
    scores = grc.join_input_layers(input_layer, None, n_layers=0, n_objects=n_objects)

    model = Model(inputs=input_layer, outputs=scores)
    model.compile(loss="mse", optimizer=grc.optimizer)
    X = np.random.randn(100, n_objects, n_features)
    y = X.sum(axis=2)
    model.fit(x=X, y=y, verbose=0)


@pytest.fixture(scope="module")
def trivial_ranking_problem():
    random_state = np.random.RandomState(123)
    x = random_state.randn(200, 5, 1)
    y_true = x.argsort(axis=1).argsort(axis=1).squeeze(axis=-1)
    return x, y_true


@pytest.mark.parametrize(
    "ranker_name, loss", zip(list(object_rankers.keys()), [0.0] * len(object_rankers))
)
def test_object_ranker_fixed(trivial_ranking_problem, ranker_name, loss):
    tf.set_random_seed(0)
    os.environ["KERAS_BACKEND"] = "tensorflow"
    np.random.seed(123)
    x, y = trivial_ranking_problem
    ranker_params = object_rankers_params[ranker_name]
    ranker_params["n_object_features"] = ranker_params["n_features"] = 1
    ranker_params["n_objects"] = 5
    ranker = object_rankers[ranker_name](**ranker_params)
    ranker.fit(x, y, epochs=100, validation_split=0, verbose=False)
    pred_scores = ranker.predict_scores(x)
    pred_loss = zero_one_rank_loss_for_scores_ties_np(y, pred_scores)
    rtol = 1e-2
    atol = 1e-4
    assert np.isclose(loss, pred_loss, rtol=rtol, atol=atol, equal_nan=False)


def test_fate_object_ranker_fixed_generator():

    def trivial_ranking_problem_generator():
        while True:
            rand = np.random.RandomState(123)
            x = rand.randn(10, 5, 1)
            y_true = x.argsort(axis=1).argsort(axis=1).squeeze(axis=-1)
            yield x, y_true

    fate = FATEObjectRanker(
        n_object_features=1,
                            n_hidden_joint_layers=1,
                            n_hidden_set_layers=1,
                            n_hidden_joint_units=5,
                            n_hidden_set_units=5,
                            kernel_regularizer=l2(1e-4),
        optimizer=optimizer,
    )
    fate.fit_generator(
        generator=trivial_ranking_problem_generator(),
        epochs=1,
        validation_split=0,
        verbose=False,
        steps_per_epoch=10,
    )
