import copy

import numpy as np
import pytest

from rl_functions.updates import (
    IterativeControl,
    OnPolicyMonteCarloControl,
    SarsaControl,
    QLearningControl,
)

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

def test_on_policy_monte_carlo_fails_when_trajectory_and_rewards_have_different_length():
    trajectory = [
        (1, 1),
        (2, 0),
        (3, 2),
    ]
    rewards = [5., -2.]
    q = {
        1: {1: 0.0},
        2: {0: 1.0},
        3: {2: 2.0},
    }
    counts = {
        1: {1: 1},
        2: {0: 1},
        3: {2: 1},
    }
    gamma = 0.0
    control = OnPolicyMonteCarloControl(gamma, q=q, counts=counts)
    with pytest.raises(Exception) as excinfo:
        control.update(trajectory, rewards)
    assert "Trajectory and rewards have differing lengths of 3 and 2, respectively" in str(excinfo.value)

def test_on_policy_monte_carlo_gives_short_sighted_results_when_gamma_is_zero():
    trajectory = [
        (1, 1),
        (2, 0),
        (3, 2),
    ]
    rewards = [5., -2., 3.]
    q = {
        1: {1: 0.0},
        2: {0: 1.0},
        3: {2: 2.0},
    }
    counts = {
        1: {1: 1},
        2: {0: 1},
        3: {2: 1},
    }
    gamma = 0.0
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
    control = OnPolicyMonteCarloControl(gamma, q=q, counts=counts)
    control.update(trajectory, rewards)
    actual_q = control.get_q()
    actual_counts = control.get_counts()
    assert expected_q == actual_q
    assert expected_counts == actual_counts

def test_on_policy_monte_carlo_gives_intermediate_results_when_gamma_is_one_half():
    trajectory = [
        (1, 1),
        (2, 0),
        (3, 2),
    ]
    rewards = [5., -2., 3.]
    q = {
        1: {1: 0.0},
        2: {0: 1.0},
        3: {2: 2.0},
    }
    counts = {
        1: {1: 1},
        2: {0: 1},
        3: {2: 1},
    }
    gamma = 0.5
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
    control = OnPolicyMonteCarloControl(gamma, q=q, counts=counts)
    control.update(trajectory, rewards)
    actual_q = control.get_q()
    actual_counts = control.get_counts()
    assert expected_q == actual_q
    assert expected_counts == actual_counts

def test_on_policy_monte_carlo_gives_long_sighted_results_when_gamma_is_one():
    trajectory = [
        (1, 1),
        (2, 0),
        (3, 2),
    ]
    rewards = [5., -2., 3.]
    q = {
        1: {1: 0.0},
        2: {0: 1.0},
        3: {2: 2.0},
    }
    counts = {
        1: {1: 1},
        2: {0: 1},
        3: {2: 1},
    }
    gamma = 1.0
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
    control = OnPolicyMonteCarloControl(gamma, q=q, counts=counts)
    control.update(trajectory, rewards)
    q = control.get_q()
    counts = control.get_counts()
    actual_q = q
    actual_counts = counts
    assert expected_q == actual_q
    assert expected_counts == actual_counts

def test_on_policy_monte_carlo_adds_actions_to_states_in_q_and_count_when_the_state_exists_but_the_state_action_pair_doesnt():
    trajectory = [
        (1, 1),
        (2, 0),
        (3, 2),
    ]
    rewards = [5., -2., 3.]
    q = {
        1: {1: 0.0},
        2: {0: 1.0},
        3: {},
    }
    counts = {
        1: {1: 1},
        2: {0: 1},
        3: {},
    }
    gamma = 0.0
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
    control = OnPolicyMonteCarloControl(gamma, q=q, counts=counts)
    control.update(trajectory, rewards)
    actual_q = control.get_q()
    actual_counts = control.get_counts()
    assert expected_q == actual_q
    assert expected_counts == actual_counts

def test_on_policy_monte_carlo_adds_state_action_pair_in_q_and_count_when_state_doesnt_exist():
    trajectory = [
        (1, 1),
        (2, 0),
        (3, 2),
    ]
    rewards = [5., -2., 3.]
    q = {
        1: {1: 0.0},
        2: {0: 1.0},
    }
    counts = {
        1: {1: 1},
        2: {0: 1},
    }
    gamma = 0.0
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
    control = OnPolicyMonteCarloControl(gamma, q=q, counts=counts)
    control.update(trajectory, rewards)
    actual_q = control.get_q()
    actual_counts = control.get_counts()
    assert expected_q == actual_q
    assert expected_counts == actual_counts

def test_sarsa_throws_exception_when_reward_for_current_step_has_already_been_decided():
    trajectory = [
        (1, 3),
        (2, 0),
    ]
    rewards = [0., 1.]
    q = {
        1: {3: 10.},
        2: {0: 20.},
    }
    alpha = 0.
    gamma = 0.5
    control = SarsaControl(alpha, gamma, q=q)
    with pytest.raises(Exception) as excinfo:
        control.update(trajectory, rewards)
    assert "Length of trajectory and rewards lists are the same; current state-action pair shouldn't yet have a reward when doing Sarsa" in str(excinfo.value)

def test_sarsa_throws_exception_when_previous_state_is_not_in_q():
    trajectory = [
        (1, 3),
        (2, 0),
    ]
    rewards = [0]
    q = {
        2: {0: 20.},
    }
    alpha = 0.5
    gamma = 0.5
    control = SarsaControl(alpha, gamma, q=q)
    with pytest.raises(Exception) as excinfo:
        control.update(trajectory, rewards)
    assert "previous state 1 not found in q function" in str(excinfo.value)

def test_sarsa_throws_exception_when_previous_action_is_not_in_q():
    trajectory = [
        (1, 3),
        (2, 0),
    ]
    rewards = [4.]
    q = {
        1: {},
        2: {0: 20.},
    }
    alpha = 0.5
    gamma = 0.5
    control = SarsaControl(alpha, gamma, q=q)
    with pytest.raises(Exception) as excinfo:
        control.update(trajectory, rewards)
    assert "previous action 3 not found in q function" in str(excinfo.value)

def test_sarsa_throws_exception_when_current_state_is_not_in_q():
    trajectory = [
        (1, 3),
        (2, 0),
    ]
    rewards = [4.]
    q = {
        1: {3: 10.},
    }
    alpha = 0.5
    gamma = 0.5
    control = SarsaControl(alpha, gamma, q=q)
    with pytest.raises(Exception) as excinfo:
        control.update(trajectory, rewards)
    assert "current state 2 not found in q function" in str(excinfo.value)

def test_sarsa_throws_exception_when_next_action_is_not_in_q():
    trajectory = [
        (1, 3),
        (2, 0),
    ]
    rewards = [4.]
    q = {
        1: {3: 10.},
        2: {},
    }
    alpha = 0.5
    gamma = 0.5
    control = SarsaControl(alpha, gamma, q=q)
    with pytest.raises(Exception) as excinfo:
        control.update(trajectory, rewards)
    assert "current action 0 not found in q function" in str(excinfo.value)

def test_sarsa_has_no_effect_on_q_when_alpha_is_zero():
    trajectory = [
        (1, 3),
        (2, 0),
    ]
    rewards = [4.]
    q = {
        1: {3: 10.},
        2: {0: 20.},
    }
    alpha = 0.
    gamma = 0.5
    expected = copy.deepcopy(q)
    control = SarsaControl(alpha, gamma, q=q)
    control.update(trajectory, rewards)
    actual = control.get_q()
    assert expected == actual

def test_sarsa_replaces_previous_value_with_reward_when_alpha_is_one_and_gamma_is_zero():
    trajectory = [
        (1, 3),
        (2, 0),
    ]
    rewards = [4.]
    q = {
        1: {3: 10.},
        2: {0: 20.},
    }
    alpha = 1.
    gamma = 0.
    expected = rewards[-1]
    control = SarsaControl(alpha, gamma, q=q)
    control.update(trajectory, rewards)
    q = control.get_q()
    s, a = trajectory[-2]
    actual = q[s][a]
    assert expected == actual

def test_sarsa_replaces_previous_value_with_reward_plus_next_value_when_alpha_is_one_and_gamma_is_one():
    trajectory = [
        (1, 3),
        (2, 0),
    ]
    rewards = [4.]
    q = {
        1: {3: 10.},
        2: {0: 20.},
    }
    alpha = 1.
    gamma = 1.
    s_p, a_p = trajectory[-1]
    expected = rewards[-1] + q[s_p][a_p]
    control = SarsaControl(alpha, gamma, q=q)
    control.update(trajectory, rewards)
    q = control.get_q()
    s, a = trajectory[-2]
    actual = q[s][a]
    assert expected == actual

def test_sarsa_gives_expected_value_when_parameters_are_valid():
    trajectory = [
        (1, 3),
        (2, 0),
    ]
    rewards = [4.]
    q = {
        1: {3: 10.},
        2: {0: 20.},
    }
    alpha = 0.3
    gamma = 0.2
    s, a = trajectory[-2]
    s_p, a_p = trajectory[-1]
    expected = (alpha * rewards[-1]
         + alpha * gamma * q[s_p][a_p]
         + (1 - alpha) * q[s][a])
    control = SarsaControl(alpha, gamma, q=q)
    control.update(trajectory, rewards)
    q = control.get_q()
    actual = q[s][a]
    assert expected == actual

def test_q_learning_throws_exception_when_reward_for_current_step_has_already_been_decided():
    trajectory = [
        (1, 3),
        (2,),
    ]
    rewards = [0., 1.]
    q = {
        1: {3: 10.},
        2: {0: 20.},
    }
    alpha = 0.
    gamma = 0.5
    control = QLearningControl(alpha, gamma, q=q)
    with pytest.raises(Exception) as excinfo:
        control.update(trajectory, rewards)
    assert "Length of trajectory and rewards lists are the same; current state shouldn't yet have a reward when doing Q-learning" in str(excinfo.value)

def test_q_learning_throws_exception_when_previous_state_is_not_in_q():
    trajectory = [
        (1, 3),
        (2,),
    ]
    rewards = [0]
    q = {
        2: {0: 20.},
    }
    alpha = 0.5
    gamma = 0.5
    control = QLearningControl(alpha, gamma, q=q)
    with pytest.raises(Exception) as excinfo:
        control.update(trajectory, rewards)
    assert "previous state 1 not found in q function" in str(excinfo.value)

def test_q_learning_throws_exception_when_previous_action_is_not_in_q():
    trajectory = [
        (1, 3),
        (2,),
    ]
    rewards = [4.]
    q = {
        1: {},
        2: {0: 20.},
    }
    alpha = 0.5
    gamma = 0.5
    control = QLearningControl(alpha, gamma, q=q)
    with pytest.raises(Exception) as excinfo:
        control.update(trajectory, rewards)
    assert "previous action 3 not found in q function" in str(excinfo.value)

def test_q_learning_throws_exception_when_current_state_is_not_in_q():
    trajectory = [
        (1, 3),
        (2,),
    ]
    rewards = [4.]
    q = {
        1: {3: 10.},
    }
    alpha = 0.5
    gamma = 0.5
    control = QLearningControl(alpha, gamma, q=q)
    with pytest.raises(Exception) as excinfo:
        control.update(trajectory, rewards)
    assert "current state 2 not found in q function" in str(excinfo.value)

def test_q_learning_throws_exception_when_current_state_has_an_associated_action():
    trajectory = [
        (1, 3),
        (2, 0),
    ]
    rewards = [4.]
    q = {
        1: {3: 10.},
    }
    alpha = 0.5
    gamma = 0.5
    control = QLearningControl(alpha, gamma, q=q)
    with pytest.raises(Exception) as excinfo:
        control.update(trajectory, rewards)
    assert "current state shouldn't yet have an action when doing Q-learning" in str(excinfo.value)

def test_q_learning_has_no_effect_on_q_when_alpha_is_zero():
    trajectory = [
        (1, 3),
        (2,),
    ]
    rewards = [4.]
    q = {
        1: {3: 10.},
        2: {0: 20.},
    }
    alpha = 0.
    gamma = 0.5
    expected = copy.deepcopy(q)
    control = QLearningControl(alpha, gamma, q=q)
    control.update(trajectory, rewards)
    actual = control.get_q()
    assert expected == actual

def test_q_learning_replaces_previous_value_with_reward_when_alpha_is_one_and_gamma_is_zero():
    trajectory = [
        (1, 3),
        (2,),
    ]
    rewards = [4.]
    q = {
        1: {3: 10.},
        2: {0: 20.},
    }
    alpha = 1.
    gamma = 0.
    expected = rewards[-1]
    control = QLearningControl(alpha, gamma, q=q)
    control.update(trajectory, rewards)
    q = control.get_q()
    s, a = trajectory[-2]
    actual = q[s][a]
    assert expected == actual

def test_q_learning_replaces_previous_value_with_reward_plus_optimal_value_when_alpha_is_one_and_gamma_is_one():
    trajectory = [
        (1, 3),
        (2,),
    ]
    rewards = [4.]
    q = {
        1: {3: 10.},
        2: {0: 20., 7: 300.},
    }
    alpha = 1.
    gamma = 1.
    expected = rewards[-1] + q[2][7]
    control = QLearningControl(alpha, gamma, q=q)
    control.update(trajectory, rewards)
    q = control.get_q()
    s, a = trajectory[-2]
    actual = q[s][a]
    assert expected == actual

def test_q_learning_gives_expected_value_when_parameters_are_valid():
    trajectory = [
        (1, 3),
        (2,),
    ]
    rewards = [4.]
    q = {
        1: {3: 10.},
        2: {0: 20., 7: 300.},
    }
    alpha = 0.3
    gamma = 0.2
    s, a = trajectory[-2]
    expected = (alpha * rewards[-1]
         + alpha * gamma * q[2][7]
         + (1 - alpha) * q[s][a])
    control = QLearningControl(alpha, gamma, q=q)
    control.update(trajectory, rewards)
    q = control.get_q()
    actual = q[s][a]
    assert expected == actual
