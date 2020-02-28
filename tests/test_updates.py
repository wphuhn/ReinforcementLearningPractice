import copy

import numpy as np

from rl_functions.updates import update_q_function

def test_update_q_function_returns_expected_results_when_it_is_called():
    prior = np.array([20., 10., 5., 19., 6.])
    action = 2
    alpha = 0.5
    reward = -4.
    expected = np.array([20., 10., 0.5, 19., 6.])
    actual = update_q_function(prior, action, alpha, reward)
    assert all(expected == actual)

def test_q_function_does_not_change_when_it_is_supplied_to_update_q_function():
    actual = np.array([20., 10., 5., 19., 6.])
    action = 2
    alpha = 0.5
    reward = -4.
    expected = copy.deepcopy(actual)
    _ = update_q_function(actual, action, alpha, reward)
    assert all(expected == actual)
