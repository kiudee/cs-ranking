import os

import numpy as np
import pytest
import tensorflow as tf
from keras.optimizers import SGD

from csrank.dataset_reader.discretechoice.util import convert_to_label_encoding
from csrank.discretechoice import *
from csrank.experiments.constants import *
from csrank.experiments.util import metrics_on_predictions
from csrank.metrics_np import categorical_accuracy_np, topk_categorical_accuracy_np, subset_01_loss

metrics = {'CategoricalAccuracy': categorical_accuracy_np, 'CategoricalTopK2': topk_categorical_accuracy_np(k=2)}
optimizer = SGD(lr=1e-3, momentum=0.9, nesterov=True)


def get_vals(values=[1.0, 1.0]):
    return dict(zip(metrics.keys(), values))


discrete_choice_functions = {
    FETA_DC: (FETADiscreteChoiceFunction, {"optimizer": optimizer}, get_vals([0.882, 0.988])),
    RANKNET_DC: (RankNetDiscreteChoiceFunction, {"optimizer": optimizer}, get_vals([0.956, 0.996])),
    CMPNET_DC: (CmpNetDiscreteChoiceFunction, {"optimizer": optimizer}, get_vals([0.994, 1.0])),
    FATE_DC: (FATEDiscreteChoiceFunction, {"n_hidden_joint_layers": 1, "n_hidden_set_layers": 1,
                                           "n_hidden_joint_units": 5, "n_hidden_set_units": 5, "optimizer": optimizer},
              get_vals([0.95, 0.998])),
    MNL: (MultinomialLogitModel, {}, get_vals([0.998, 1.0])),
    NLM: (NestedLogitModel, {}, get_vals()),
    PCL: (PairedCombinatorialLogit, {}, get_vals()),
    GEV: (GeneralizedNestedLogitModel, {}, get_vals()),
    RANKSVM_DC: (PairwiseSVMDiscreteChoiceFunction, {}, get_vals([0.982, 998]))
}


@pytest.fixture(scope="module")
def trivial_discrete_choice_problem():
    random_state = np.random.RandomState(42)
    x = random_state.randn(500, 5, 2)
    w = random_state.rand(2)
    y_true = np.argmax(np.dot(x, w), axis=1)
    y_true = convert_to_label_encoding(y_true, 5)
    return x, y_true


@pytest.mark.parametrize("name", list(discrete_choice_functions.keys()))
def test_choice_function_fixed(trivial_discrete_choice_problem, name):
    tf.set_random_seed(0)
    os.environ["KERAS_BACKEND"] = "tensorflow"
    np.random.seed(123)
    x, y = trivial_discrete_choice_problem
    choice_function = discrete_choice_functions[name][0]
    params, accuracies = discrete_choice_functions[name][1], discrete_choice_functions[name][2]
    params["n_objects"], params["n_object_features"] = tuple(x.shape[1:])
    learner = choice_function(**params)
    learner.fit(x, y, epochs=100, validation_split=0, verbose=False)
    s_pred = learner.predict_scores(x)
    y_pred = learner.predict_for_scores(s_pred)
    y_pred_2 = learner.predict(x)
    rtol = 1e-2
    atol = 1e-4
    assert np.isclose(0.0, subset_01_loss(y_pred, y_pred_2), rtol=rtol, atol=atol, equal_nan=False)
    for key, value in accuracies.items():
        metric = metrics[key]
        if metric in metrics_on_predictions:
            pred_loss = metric(y, y_pred)
        else:
            pred_loss = metric(y, s_pred)
        assert np.isclose(value, pred_loss, rtol=rtol, atol=atol, equal_nan=False)
