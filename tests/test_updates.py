import copy

import numpy as np

from rl_functions.updates import update_q_function

def test_update_q_function_updates_existing_state_action_pair_when_the_pair_is_in_the_q_function():
    state = 0
    action = 2
    old_value = 5.
    prior = {
        state: {
            0: 20.,
            1: 10.,
            action: old_value,
            3: 19.,
            4: 6.,
        },
        1: {
            0: 25.,
            1: 15.,
            2: 0.,
            3: 14.,
            4: 11.,
        },
    }
    alpha = 0.5
    reward = -4.
    new_value = old_value + alpha * (reward - old_value)
    expected = copy.deepcopy(prior)
    expected[state][action] = new_value
    actual = update_q_function(prior, state, action, alpha, reward)
    assert expected == actual

def test_update_q_function_adds_new_state_action_pair_when_the_state_is_not_in_the_q_function():
    prior = {
        0: {
            0: 20.,
            1: 10.,
            2: 5.,
            3: 19.,
            4: 6.,
        },
        1.: {
            0: 25.,
            1: 15.,
            2: 0.,
            3: 14.,
            4: 11.,
        },
    }
    state = 2
    action = 7
    alpha = 0.5
    reward = -4.
    new_value = alpha * reward
    expected = copy.deepcopy(prior)
    expected[state] = {}
    expected[state][action] = new_value
    actual = update_q_function(prior, state, action, alpha, reward)
    assert expected == actual

def test_update_q_function_adds_new_action_to_existing_state_with_optimistic_value_when_the_action_is_not_associated_with_the_state():
    state = 1
    max_value = 25.
    prior = {
        0: {
            0: 20.,
            1: 10.,
            2: 5.,
            3: 19.,
            4: 6.,
        },
        state: {
            0: max_value,
            1: 15.,
            2: 0.,
            3: 14.,
            4: 11.,
        },
    }
    action = 7
    alpha = 0.5
    reward = -4.
    new_value = max_value + alpha * (reward - max_value)
    expected = copy.deepcopy(prior)
    expected[state][action] = new_value
    actual = update_q_function(prior, state, action, alpha, reward)
    assert expected == actual

def test_q_function_does_not_change_when_it_is_supplied_to_update_q_function():
    state = 0
    action = 3
    actual = {
        0: {
            0: 20.,
            1: 10.,
            2: 5.,
            3: 19.,
            4: 6.
        },
        1: {
            0: 25.,
            1: 15.,
            2: 0.,
            3: 14.,
            4: 11.,
        },
    }
    alpha = 0.5
    reward = -4.
    expected = copy.deepcopy(actual)
    _ = update_q_function(actual, state, action, alpha, reward)
    assert expected == actual
