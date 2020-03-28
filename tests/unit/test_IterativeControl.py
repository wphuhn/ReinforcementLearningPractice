import copy

import numpy as np
import pytest

from rl_functions.controls import IterativeControl

from utilities import create_rngs_with_fixed_seed

def test_update_iterative_updates_existing_state_action_pair_when_the_pair_is_in_the_q_function():
    state = 0
    action = 2
    old_value = 5.
    trajectory = [(state, action)]
    q = {
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
    rewards = [reward]
    expected = old_value + alpha * (reward - old_value)
    control = IterativeControl(alpha, q=q)
    control.update(trajectory, rewards)
    q = control.get_q()
    actual = q[state][action]
    assert expected == actual

def test_update_iterative_adds_new_state_action_pair_when_the_state_is_not_in_the_q_function():
    q = {
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
    trajectory = [(state, action)]
    alpha = 0.5
    reward = -4.
    rewards = [reward]
    expected = alpha * reward
    control = IterativeControl(alpha, q=q)
    control.update(trajectory, rewards)
    q = control.get_q()
    actual = q[state][action]
    assert expected == actual

def test_update_iterative_adds_new_action_to_existing_state_with_optimistic_value_when_the_action_is_not_associated_with_the_state():
    state = 1
    max_value = 25.
    q = {
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
    trajectory = [(state, action)]
    alpha = 0.5
    reward = -4.
    rewards = [reward]
    expected = max_value + alpha * (reward - max_value)
    control = IterativeControl(alpha, q=q)
    control.update(trajectory, rewards)
    q = control.get_q()
    actual = q[state][action]
    assert expected == actual
