import logging
from abc import ABCMeta

import numpy as np
from keras.layers import Input
from keras.models import Model

from ..fate_ranking import FATERankingCore
from ..util import tunable_parameters_ranges


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
