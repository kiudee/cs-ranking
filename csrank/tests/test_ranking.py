import logging
import os
from abc import ABCMeta

import numpy as np
import pytest
import tensorflow as tf
from keras.layers import Input
from keras.models import Model
from keras.optimizers import SGD
from keras.regularizers import l2

from csrank import FETANetwork, RankNet, CmpNet, ExpectedRankRegression, RankSVM
from ..fate_ranking import FATERankingCore, FATEObjectRanker
from ..util import tunable_parameters_ranges, zero_one_rank_loss_for_scores_ties_np

RANKSVM = 'ranksvm'
ERR = 'err'
CMPNET = "cmpnet"
RANKNET = 'ranknet'
FETA_RANKER = 'feta_ranker'
FATE_RANKER = "fate_ranker"

object_rankers = {FETA_RANKER: FETANetwork, RANKNET: RankNet, CMPNET: CmpNet,
                  ERR: ExpectedRankRegression, RANKSVM: RankSVM,
                  FATE_RANKER: FATEObjectRanker}
object_rankers_params = {
    FETA_RANKER: {"add_zeroth_order_model": True, "optimizer": SGD(lr=1e-3, momentum=0.9, nesterov=True)},
    RANKNET: {"optimizer": SGD(lr=1e-3, momentum=0.9, nesterov=True)},
    CMPNET: {"optimizer": SGD(lr=1e-3, momentum=0.9, nesterov=True)},
    FATE_RANKER: {"n_hidden_joint_layers": 1, "n_hidden_set_layers": 1, "n_hidden_joint_units": 5,
                  "n_hidden_set_units": 5, "optimizer": SGD(lr=1e-3, momentum=0.9, nesterov=True)},
    ERR: {}, RANKSVM: {}}


def test_construction_core():
    n_objects = 3
    n_features = 2

    # Create mock class:
    class MockClass(FATERankingCore, metaclass=ABCMeta):
        def set_tunable_parameters(self, point):
            super().set_tunable_parameters(point)

        def predict(self, *args, **kwargs):
            pass

        def fit(self, *args, **kwargs):
            pass

        def set_tunable_parameter_ranges(cls, param_ranges_dict):
            logger = logging.getLogger("MockClass")
            logger.info("Setting parameter ranges: " + repr(param_ranges_dict))
            return tunable_parameters_ranges(cls, logger, param_ranges_dict)

        def tunable_parameters(cls):
            if cls._tunable is None:
                super().tunable_parameters()

    grc = MockClass(n_objects=n_objects, n_features=n_features)
    grc._construct_layers(activation=grc.activation,
                          kernel_initializer=grc.kernel_initializer,
                          kernel_regularizer=grc.kernel_regularizer)
    input_layer = Input(shape=(n_objects, n_features))
    scores = grc.join_input_layers(input_layer,
                                   None,
                                   n_layers=0,
                                   n_objects=n_objects)

    model = Model(inputs=input_layer, outputs=scores)
    model.compile(loss='mse', optimizer=grc.optimizer)
    X = np.random.randn(100, n_objects, n_features)
    y = X.sum(axis=2)
    model.fit(x=X, y=y, verbose=0)


@pytest.fixture(scope="module")
def trivial_ranking_problem():
    rand = np.random.RandomState(123)
    x = rand.randn(100, 5, 1)
    y_true = x.argsort(axis=1).argsort(axis=1).squeeze(axis=-1)
    return x, y_true


def test_fate_object_ranker_fixed(trivial_ranking_problem):
    tf.set_random_seed(0)
    os.environ["KERAS_BACKEND"] = "tensorflow"
    np.random.seed(123)
    for ranker_name in object_rankers.keys():
        loss = 0.0
        rtol = 1e-2
        atol = 1e-8
        if ranker_name == ERR:
            loss = 0.5
            rtol = 1e-2
            atol = 1e-2
        if ranker_name == RANKSVM:
            rtol = 1e-2
            atol = 1e-2
        assert object_ranker_fixed(trivial_ranking_problem, ranker_name=ranker_name, loss=loss, rtol=rtol, atol=atol)


def object_ranker_fixed(trivial_ranking_problem, ranker_name=FATE_RANKER, loss=0.0, rtol=1e-2,
                        atol=1e-8):
    x, y = trivial_ranking_problem
    ranker_params = object_rankers_params[ranker_name]
    ranker_params['n_object_features'] = ranker_params['n_features'] = 1
    ranker_params['n_objects'] = 5
    ranker = object_rankers[ranker_name](**ranker_params)
    ranker.fit(x, y, epochs=50, validation_split=0, verbose=False)
    pred_scores = ranker.predict_scores(x)
    pred_loss = zero_one_rank_loss_for_scores_ties_np(y, pred_scores)
    print("ranker : {} and 0/1 pred loss: {}".format(ranker_name, pred_loss))
    return np.isclose(loss, pred_loss, rtol=rtol, atol=atol, equal_nan=False)


def test_fate_object_ranker_fixed_generator():
    def trivial_ranking_problem_generator():
        while True:
            rand = np.random.RandomState(123)
            x = rand.randn(10, 5, 1)
            y_true = x.argsort(axis=1).argsort(axis=1).squeeze(axis=-1)
            yield x, y_true

    fate = FATEObjectRanker(n_object_features=1,
                            n_hidden_joint_layers=1,
                            n_hidden_set_layers=1,
                            n_hidden_joint_units=5,
                            n_hidden_set_units=5,
                            kernel_regularizer=l2(1e-4),
                            optimizer=SGD(lr=1e-3, momentum=0.9, nesterov=True))
    fate.fit_generator(generator=trivial_ranking_problem_generator(),
                       epochs=1, validation_split=0, verbose=False,
                       steps_per_epoch=10)
