import copy

import numpy as np
import pytest

from rl_functions.controls import OnPolicyMonteCarloControl

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
