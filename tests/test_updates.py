import copy

import numpy as np
import pytest

from rl_functions.updates import (
    update_q_iterative,
    update_q_on_policy_monte_carlo,
    update_q_sarsa,
)

def test_update_q_iterative_updates_existing_state_action_pair_when_the_pair_is_in_the_q_function():
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
    actual = update_q_iterative(prior, state, action, alpha, reward)
    assert expected == actual

def test_update_q_iterative_adds_new_state_action_pair_when_the_state_is_not_in_the_q_function():
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
    actual = update_q_iterative(prior, state, action, alpha, reward)
    assert expected == actual

def test_update_q_iterative_adds_new_action_to_existing_state_with_optimistic_value_when_the_action_is_not_associated_with_the_state():
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
    actual = update_q_iterative(prior, state, action, alpha, reward)
    assert expected == actual

def test_q_function_does_not_change_when_it_is_supplied_to_update_q_iterative():
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
    _ = update_q_iterative(actual, state, action, alpha, reward)
    assert expected == actual

def test_on_policy_monte_carlo_fails_when_trajectory_and_rewards_have_different_length():
    trajectory = [
        (1, 1),
        (2, 0),
        (3, 2),
    ]
    rewards = [5., -2.]
    prev_q = {
        1: {1: 0.0},
        2: {0: 1.0},
        3: {2: 2.0},
    }
    prev_counts = {
        1: {1: 1},
        2: {0: 1},
        3: {2: 1},
    }
    gamma = 0.0
    with pytest.raises(Exception) as excinfo:
        _, _ = update_q_on_policy_monte_carlo(
           trajectory,
           rewards,
           prev_q,
           prev_counts,
           gamma,
       )
    assert "Trajectory and rewards have differing lengths of 3 and 2, respectively" in str(excinfo.value)

def test_on_policy_monte_carlo_gives_short_sighted_results_when_gamma_is_zero():
    trajectory = [
        (1, 1),
        (2, 0),
        (3, 2),
    ]
    rewards = [5., -2., 3.]
    prev_q = {
        1: {1: 0.0},
        2: {0: 1.0},
        3: {2: 2.0},
    }
    prev_counts = {
        1: {1: 1},
        2: {0: 1},
        3: {2: 1},
    }
    gamma = 0.0
    actual_q, actual_counts = update_q_on_policy_monte_carlo(
        trajectory,
        rewards,
        prev_q,
        prev_counts,
        gamma,
    )
    expected_q = {
        1: {1: 2.5},
        2: {0: -0.5},
        3: {2: 2.5},
    }
    expected_counts = {
        1: {1: 2},
        2: {0: 2},
        3: {2: 2},
    }
    assert expected_q == actual_q
    assert expected_counts == actual_counts

def test_on_policy_monte_carlo_gives_intermediate_results_when_gamma_is_one_half():
    trajectory = [
        (1, 1),
        (2, 0),
        (3, 2),
    ]
    rewards = [5., -2., 3.]
    prev_q = {
        1: {1: 0.0},
        2: {0: 1.0},
        3: {2: 2.0},
    }
    prev_counts = {
        1: {1: 1},
        2: {0: 1},
        3: {2: 1},
    }
    gamma = 0.5
    actual_q, actual_counts = update_q_on_policy_monte_carlo(
        trajectory,
        rewards,
        prev_q,
        prev_counts,
        gamma,
    )
    expected_q = {
        1: {1: 2.375},
        2: {0: 0.25},
        3: {2: 2.5},
    }
    expected_counts = {
        1: {1: 2},
        2: {0: 2},
        3: {2: 2},
    }
    assert expected_q == actual_q
    assert expected_counts == actual_counts

def test_on_policy_monte_carlo_gives_long_sighted_results_when_gamma_is_one():
    trajectory = [
        (1, 1),
        (2, 0),
        (3, 2),
    ]
    rewards = [5., -2., 3.]
    prev_q = {
        1: {1: 0.0},
        2: {0: 1.0},
        3: {2: 2.0},
    }
    prev_counts = {
        1: {1: 1},
        2: {0: 1},
        3: {2: 1},
    }
    gamma = 1.0
    actual_q, actual_counts = update_q_on_policy_monte_carlo(
        trajectory,
        rewards,
        prev_q,
        prev_counts,
        gamma,
    )
    expected_q = {
        1: {1: 3.0},
        2: {0: 1.0},
        3: {2: 2.5},
    }
    expected_counts = {
        1: {1: 2},
        2: {0: 2},
        3: {2: 2},
    }
    assert expected_q == actual_q
    assert expected_counts == actual_counts

def test_on_policy_monte_carlo_adds_actions_to_states_in_q_and_count_when_the_state_exists_but_the_state_action_pair_doesnt():
    trajectory = [
        (1, 1),
        (2, 0),
        (3, 2),
    ]
    rewards = [5., -2., 3.]
    prev_q = {
        1: {1: 0.0},
        2: {0: 1.0},
        3: {},
    }
    prev_counts = {
        1: {1: 1},
        2: {0: 1},
        3: {},
    }
    gamma = 0.0
    actual_q, actual_counts = update_q_on_policy_monte_carlo(
        trajectory,
        rewards,
        prev_q,
        prev_counts,
        gamma,
    )
    expected_q = {
        1: {1: 2.5},
        2: {0: -0.5},
        3: {2: 3.0},
    }
    expected_counts = {
        1: {1: 2},
        2: {0: 2},
        3: {2: 1},
    }
    assert expected_q == actual_q
    assert expected_counts == actual_counts

def test_on_policy_monte_carlo_adds_state_action_pair_in_q_and_count_when_state_doesnt_exist():
    trajectory = [
        (1, 1),
        (2, 0),
        (3, 2),
    ]
    rewards = [5., -2., 3.]
    prev_q = {
        1: {1: 0.0},
        2: {0: 1.0},
    }
    prev_counts = {
        1: {1: 1},
        2: {0: 1},
    }
    gamma = 0.0
    actual_q, actual_counts = update_q_on_policy_monte_carlo(
        trajectory,
        rewards,
        prev_q,
        prev_counts,
        gamma,
    )
    expected_q = {
        1: {1: 2.5},
        2: {0: -0.5},
        3: {2: 3.0},
    }
    expected_counts = {
        1: {1: 2},
        2: {0: 2},
        3: {2: 1},
    }
    assert expected_q == actual_q
    assert expected_counts == actual_counts

def test_sarsa_throws_exception_when_previous_state_is_not_in_q():
    prev_pair = (1, 3)
    reward = 0
    next_pair = (2, 0)
    q = {
        2: {0: 20.},
    }
    alpha = 0.5
    gamma = 0.5
    with pytest.raises(Exception) as excinfo:
        _ = update_q_sarsa(prev_pair, reward, next_pair, q, alpha, gamma)
    assert "previous state 1 not found in q function" in str(excinfo.value)

def test_sarsa_throws_exception_when_previous_action_is_not_in_q():
    prev_pair = (1, 3)
    reward = 4.
    next_pair = (2, 0)
    q = {
        1: {},
        2: {0: 20.},
    }
    alpha = 0.5
    gamma = 0.5
    with pytest.raises(Exception) as excinfo:
        _ = update_q_sarsa(prev_pair, reward, next_pair, q, alpha, gamma)
    assert "previous action 3 not found in q function" in str(excinfo.value)

def test_sarsa_throws_exception_when_next_state_is_not_in_q():
    prev_pair = (1, 3)
    reward = 4.
    next_pair = (2, 0)
    q = {
        1: {3: 10.},
    }
    alpha = 0.5
    gamma = 0.5
    with pytest.raises(Exception) as excinfo:
        _ = update_q_sarsa(prev_pair, reward, next_pair, q, alpha, gamma)
    assert "next state 2 not found in q function" in str(excinfo.value)

def test_sarsa_throws_exception_when_next_action_is_not_in_q():
    prev_pair = (1, 3)
    reward = 4.
    next_pair = (2, 0)
    q = {
        1: {3: 10.},
        2: {},
    }
    alpha = 0.5
    gamma = 0.5
    with pytest.raises(Exception) as excinfo:
        _ = update_q_sarsa(prev_pair, reward, next_pair, q, alpha, gamma)
    assert "next action 0 not found in q function" in str(excinfo.value)

def test_sarsa_throws_exception_when_next_action_is_not_in_q():
    prev_pair = (1, 3)
    reward = 4.
    next_pair = (2, 0)
    q = {
        1: {3: 10.},
        2: {},
    }
    alpha = 0.5
    gamma = 0.5
    with pytest.raises(Exception) as excinfo:
        _ = update_q_sarsa(prev_pair, reward, next_pair, q, alpha, gamma)
    assert "next action 0 not found in q function" in str(excinfo.value)

def test_sarsa_has_no_effect_on_q_when_alpha_is_zero():
    prev_pair = (1, 3)
    reward = 4.
    next_pair = (2, 0)
    expected = {
        1: {3: 10.},
        2: {0: 20.},
    }
    alpha = 0.
    gamma = 0.5
    actual = update_q_sarsa(prev_pair, reward, next_pair, expected, alpha, gamma)
    assert expected == actual

def test_sarsa_replaces_previous_value_with_reward_when_alpha_is_one_and_gamma_is_zero():
    prev_pair = (1, 3)
    reward = 4.
    next_pair = (2, 0)
    q = {
        1: {3: 10.},
        2: {0: 20.},
    }
    alpha = 1.
    gamma = 0.
    expected = reward
    q_new = update_q_sarsa(prev_pair, reward, next_pair, q, alpha, gamma)
    actual = q_new[prev_pair[0]][prev_pair[1]]
    assert expected == actual

def test_sarsa_replaces_previous_value_with_reward_plus_next_value_when_alpha_is_one_and_gamma_is_one():
    prev_pair = (1, 3)
    reward = 4.
    next_pair = (2, 0)
    q = {
        1: {3: 10.},
        2: {0: 20.},
    }
    alpha = 1.
    gamma = 1.
    expected = reward + q[next_pair[0]][next_pair[1]]
    q_new = update_q_sarsa(prev_pair, reward, next_pair, q, alpha, gamma)
    actual = q_new[prev_pair[0]][prev_pair[1]]
    assert expected == actual

def test_sarsa_gives_expected_value_when_parameters_are_valid():
    prev_pair = (1, 3)
    reward = 4.
    next_pair = (2, 0)
    q = {
        1: {3: 10.},
        2: {0: 20.},
    }
    alpha = 0.3
    gamma = 0.2
    expected = (alpha * reward
         + alpha * gamma * q[next_pair[0]][next_pair[1]]
         + (1 - alpha) * q[prev_pair[0]][prev_pair[1]])
    q_new = update_q_sarsa(prev_pair, reward, next_pair, q, alpha, gamma)
    actual = q_new[prev_pair[0]][prev_pair[1]]
    assert expected == actual
