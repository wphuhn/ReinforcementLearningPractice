import copy

import numpy as np
import pytest

from rl_functions.controls import IterativeControl
from rl_functions.fakes import FakeEnv

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

def test_run_iterative_episode_returns_error_when_negative_steps_requested():
    alpha = 0.5
    max_n_steps = -1
    control = IterativeControl(alpha)
    with pytest.raises(Exception) as excinfo:
        _, _, _ = control.run_episode(None, None, max_n_steps=max_n_steps)
    assert "Episode cannot have a negative number of steps" in str(excinfo.value)

def test_run_iterative_episode_returns_trajectory_with_initial_state_when_zero_steps_requested():
    alpha = 0.5
    control = IterativeControl(alpha)
    initial_state = 5
    max_n_steps = 0
    trajectory, rewards, infos = control.run_episode(None, initial_state, max_n_steps=max_n_steps)
    assert trajectory == [(initial_state, None)]
    assert rewards == []
    assert infos == []

def test_run_iterative_episode_stops_when_max_number_of_steps_has_been_reached():
    alpha = 0.5
    initial_q = {0: {0: 10}}
    control = IterativeControl(alpha, q=initial_q)
    max_n_steps = 3
    initial_state = 0
    env = FakeEnv(observation=initial_state, done=False)
    trajectory, rewards, infos = control.run_episode(env, initial_state, max_n_steps=max_n_steps)
    assert len(trajectory) == max_n_steps + 1
    assert len(rewards) == max_n_steps
    assert len(infos) == max_n_steps

def test_run_iterative_episode_stops_when_env_signals_terminal_state_has_been_reached():
    alpha = 0.5
    initial_q = {0: {0: 10}}
    control = IterativeControl(alpha, q=initial_q)
    # Set up fake environment that will throw the stop signal after a pre-set
    # number of iterations
    initial_state = 0
    done = [False, False, False, True]
    env = FakeEnv(observation=initial_state, done=done)
    trajectory, rewards, infos = control.run_episode(env, initial_state)
    assert len(trajectory) == len(done) + 1
    assert len(rewards) == len(done)
    assert len(infos) == len(done)

def test_run_iterative_episode_gives_expected_results():
    alpha = 0.5
    initial_state = 0
    initial_q = {
        0: {
            0: 20.,
            1: 10.,
        },
        1: {
            0: 14.,
            1: 20.,
        },
        2: {
            0: -1.,
            1: -2.,
        }
    }
    control = IterativeControl(alpha, q=initial_q)
    # Set up fake environment that will throw the stop signal after a pre-set
    # number of iterations
    states = [0, 1, 2, 1, 2]
    expected_rewards = [-40, 10, -20, 3, 5]
    done = [False, False, False, False, True]
    expected_infos = ["1", "2", "3", "4", "5"]
    env = FakeEnv(observation=states, reward=expected_rewards, done=done, info=expected_infos)
    actual_trajectory, actual_rewards, actual_infos = control.run_episode(env, initial_state)
    actual_q = control.get_q()
    expected_q = {
        0: {
            0: -10.,
            1: 10.,
        },
        1: {
            0: 9.5,
            1: 0.,
        },
        2: {
            0: 1.,
            1: -2.,
        }
    }
    expected_trajectory = [
        (0, 0),
        (0, 1),
        (1, 1),
        (2, 0),
        (1, 0),
        (2, None)
    ]
    assert expected_rewards == actual_rewards
    assert expected_infos == actual_infos
    assert expected_trajectory == actual_trajectory
    assert expected_q == actual_q
