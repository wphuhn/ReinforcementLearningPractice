import copy

import numpy as np
import pytest

from rl_functions.controls import QLearningControl

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
