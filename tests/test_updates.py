import copy

import numpy as np

from rl_functions.updates import update_q_function

def test_update_q_function_returns_expected_results_when_it_is_called():
    prior = {
        0: {
            0: 20.,
            1: 10.,
            2: 5.,
            3: 19.,
            4: 6.
        }
    }
    state = 0
    action = 2
    alpha = 0.5
    reward = -4.
    expected = {
        0: {
            0: 20.,
            1: 10.,
            2: 0.5,
            3: 19.,
            4: 6.
        }
    }
    actual = update_q_function(prior, state, action, alpha, reward)
    assert expected == actual

def test_q_function_does_not_change_when_it_is_supplied_to_update_q_function():
    actual = {
        0: {
            0: 20.,
            1: 10.,
            2: 5.,
            3: 19.,
            4: 6.
        }
    }
    state = 0
    action = 2
    alpha = 0.5
    reward = -4.
    expected = copy.deepcopy(actual)
    _ = update_q_function(actual, state, action, alpha, reward)
    assert expected == actual
