from csrank.objectranking import FATEObjectRanker


def test_get_and_set_params():
    """Tests the get_params and set_params function of our learners."""
    # FATEObjectRanker is chosen as an arbitrary example; the functions are
    # implemented in the learner superclass.
    learner = FATEObjectRanker()
    params = set(learner.get_params().keys())
    # Regular parameters
    assert "activation" in params
    assert "kernel_initializer" in params
    # Regular nested parameters
    assert "optimizer" in params
    assert "optimizer__learning_rate" in params
    # A special case of a nested parameter, since there is no base
    # "hidden_dense_layer" parameter.
    assert "hidden_dense_layer__bias_constraint" in params

    # All parameters returned by get_parameters can also be set.
    learner.set_params(batch_size=42, optimizer__learning_rate=10)

    assert learner.get_params()["batch_size"] == 42
