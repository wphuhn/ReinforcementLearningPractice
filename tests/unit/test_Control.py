import copy

import numpy as np
import pytest

from rl_functions.controls import Control

from utilities import create_rngs_with_fixed_seed

def test_iterative_control_returns_greedy_action_when_epsilon_is_zero():
    state = 1
    action = 7
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
            1: 15.,
            2: 0.,
            3: 14.,
            4: 11.,
            action: max_value,
        },
    }
    epsilon = 0.0
    control = Control(epsilon=epsilon, random_policy=None, rng=None, q=q)
    expected = action
    for _ in range(10):
        actual = control.next_action(state)
        assert expected == actual

def test_iterative_control_returns_expected_results_when_epsilon_is_non_zero():
    state = 1
    action = 7
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
            1: 15.,
            2: 0.,
            3: 14.,
            4: 11.,
            action: max_value,
        },
    }
    policy, rng = create_rngs_with_fixed_seed(10, 0, 0)
    epsilon = 0.5
    control = Control(
      epsilon=epsilon,
      random_policy=policy,
      rng=rng,
      q=q,
    )
    expected_actions = [7, 6, 5, 2, 7, 7, 7, 7, 7, 7]
    for expected in expected_actions:
        actual = control.next_action(state)
        assert expected == actual
