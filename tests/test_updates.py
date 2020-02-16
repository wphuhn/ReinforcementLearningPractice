import copy

import numpy as np

from rl_functions.updates import update_q_function

def test_update_q_function_gives_expected_results():
    q_function_prior = np.array([20., 10., 5., 19., 6.])
    action = 2
    alpha = 0.5
    reward_action = -4.
    q_function_expected = np.array([20., 10., 3., 19., 6.])
    q_function_actual = update_q_function(q_function_prior, action, alpha, reward_action)
    assert(all(q_function_expected == q_function_actual))

def test_update_q_function_does_not_change_prior():
    q_function_prior = np.array([20., 10., 5., 19., 6.])
    action = 2
    alpha = 0.5
    reward_action = -4.
    q_function_expected = copy.deepcopy(q_function_prior)
    _ = update_q_function(q_function_prior, action, alpha, reward_action)
    assert(all(q_function_expected == q_function_prior))


