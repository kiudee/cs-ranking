import os

import numpy as np
import pytest
import tensorflow as tf
from keras.optimizers import SGD

from csrank.choicefunction import *
from csrank.experiments.constants import *
from csrank.experiments.util import metrics_on_predictions
from csrank.metrics_np import f1_measure, subset_01_loss, instance_informedness, auc_score
from csrank.tests.test_ranking import check_leaner

choice_metrics = {'F1Score': f1_measure, 'Informedness': instance_informedness, "AucScore": auc_score}
optimizer = SGD(lr=1e-3, momentum=0.9, nesterov=True)


def get_vals(values):
    return dict(zip(choice_metrics.keys(), values))


choice_functions = {
    FETA_CHOICE: (FETAChoiceFunction, {"add_zeroth_order_model": True, "optimizer": optimizer},
                  get_vals([0.946, 0.9684, 0.9998])),
    FATE_CHOICE: (FATEChoiceFunction, {"n_hidden_joint_layers": 1, "n_hidden_set_layers": 1, "n_hidden_joint_units": 5,
                                       "n_hidden_set_units": 5, "optimizer": optimizer},
                  get_vals([0.8185, 0.6845, 0.9924])),
    FATELINEAR_CHOICE: (FATELinearChoiceFunction, {}, get_vals([0.8014, 0.4906, 0.9998])),
    FETALINEAR_CHOICE: (FETALinearChoiceFunction, {}, get_vals([0.8782, 0.8894, 0.9998])),
    RANKNET_CHOICE: (RankNetChoiceFunction, {"optimizer": optimizer}, get_vals([0.9522, 0.9866, 1.0])),
    CMPNET_CHOICE: (CmpNetChoiceFunction, {"optimizer": optimizer}, get_vals([0.8554, 0.8649, 0.966])),
    GLM_CHOICE: (GeneralizedLinearModel, {}, get_vals([0.9567, 0.9955, 1.0])),
    RANKSVM_CHOICE: (PairwiseSVMChoiceFunction, {}, get_vals([0.9522, 0.9955, 1.0]))
}


@pytest.fixture(scope="module")
def trivial_choice_problem():
    random_state = np.random.RandomState(42)
    x = random_state.randn(200, 5, 1)
    y_true = np.array(x.squeeze(axis=-1) > np.mean(x))
    return x, y_true


@pytest.mark.parametrize("name", list(choice_functions.keys()))
def test_choice_function_fixed(trivial_choice_problem, name):
    tf.set_random_seed(0)
    os.environ["KERAS_BACKEND"] = "tensorflow"
    np.random.seed(123)
    x, y = trivial_choice_problem
    choice_function = choice_functions[name][0]
    params, accuracies = choice_functions[name][1], choice_functions[name][2]
    params["n_objects"], params["n_object_features"] = tuple(x.shape[1:])
    learner = choice_function(**params)
    if "linear" in name:
        learner.fit(x, y, epochs=10, validation_split=0, verbose=False)
    else:
        learner.fit(x, y, epochs=100, validation_split=0, verbose=False)
    s_pred = learner.predict_scores(x)
    y_pred = learner.predict_for_scores(s_pred)
    y_pred_2 = learner.predict(x)
    rtol = 1e-2
    atol = 5e-2
    assert np.isclose(0.0, subset_01_loss(y_pred, y_pred_2), rtol=rtol, atol=atol, equal_nan=False)
    for key, value in accuracies.items():
        metric = choice_metrics[key]
        if metric in metrics_on_predictions:
            pred_loss = metric(y, y_pred)
        else:
            pred_loss = metric(y, s_pred)
        assert np.isclose(value, pred_loss, rtol=rtol, atol=atol, equal_nan=False)
    params = {"n_hidden": 20, "n_units": 20, "n_hidden_set_units": 2, "n_hidden_set_layers": 10,
              "n_hidden_joint_units": 2, "n_hidden_joint_layers": 10, "reg_strength": 1e-3, "learning_rate": 1e-1,
              "batch_size": 32, "alpha": 0.5, "l1_ratio": 0.7, "tol": 1e-2, "C": 10, "n_mixtures": 10, "n_nests": 5,
              "regularization": "l2"}
    learner.set_tunable_parameters(**params)
    check_leaner(learner, params, rtol, atol)
