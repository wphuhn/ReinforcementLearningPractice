import copy

import numpy as np
from numpy.random import default_rng
import pytest

from rl_functions.policies import GreedyPolicy

from constants import N_CHOICES
from utilities import create_random_policy_with_fixed_rng

def test_creating_greedy_policy_throws_exception_when_epsilon_is_greater_than_zero_and_random_policy_not_provided():
    with pytest.raises(Exception) as excinfo:
        _ = GreedyPolicy(epsilon=1.0)
    assert "when specifying an epsilon value greater than 0, you must also provide a random policy!" in str(excinfo.value)

def test_greedy_policy_picks_unique_greedy_answer_when_there_is_a_clear_greedy_maximum():
    q_function = np.array([17.1, -5.5, 20.2, -20.9, 5.1, 1.0, -1.0, 1.5])
    greedy_policy = GreedyPolicy()
    action_expected = 2
    for _ in range(100):
        action_actual = greedy_policy.next_action(q_function)
        assert(action_expected == action_actual)

def test_epsilon_greedy_policy_picks_unique_greedy_answer_when_there_is_a_unique_greedy_maximum_and_epsilon_is_zero():
    q_function = np.array([17.1, -5.5, 20.2, -20.9, 5.1, 1.0, -1.0, 1.5])
    greedy_policy = GreedyPolicy(epsilon=0.0)
    action_expected = 2
    for _ in range(100):
        action_actual = greedy_policy.next_action(q_function)
        assert(action_expected == action_actual)

def test_greedy_policy_breaks_ties_in_a_pseudorandom_fashion_when_there_is_more_than_one_greedy_maximum():
    q_function = np.array([20, 20, 20, 20, 20, 20, 20, 20])
    random_generator = default_rng(seed=0)
    greedy_policy = GreedyPolicy(random_generator=random_generator)
    actions_expected = [6, 5, 4, 2, 2]
    for action_expected in actions_expected:
        action_actual = greedy_policy.next_action(q_function)
        assert(action_expected == action_actual)

def test_epsilon_greedy_policy_gives_expected_results():
    random_policy = create_random_policy_with_fixed_rng(N_CHOICES, 0)
    random_generator = default_rng(seed=0)
    greedy_policy = GreedyPolicy(epsilon=0.5, random_policy=random_policy, random_generator=random_generator)
    q_function = np.array([17.1, -5.5, 20.2, -20.9, 5.1, 1.0, -1.0, 1.5])
    actions_expected = [2, 5, 4, 2, 2, 2]
    for action_expected in actions_expected:
        action_actual = greedy_policy.next_action(q_function)
        assert(action_expected == action_actual)

def test_epsilon_greedy_policy_gives_identical_results_to_greedy_policy_when_there_is_a_unique_greedy_maximum_and_epsilon_equals_zero():
    q_function = np.array([17.1, -5.5, 20.2, -20.9, 5.1, 1.0, -1.0, 1.5])
    random_generator = default_rng(seed=0)
    greedy_policy = GreedyPolicy()
    epsilon_greedy_policy = GreedyPolicy(epsilon=0.0)
    for _ in range(100):
        action_expected = greedy_policy.next_action(q_function)
        action_actual = epsilon_greedy_policy.next_action(q_function)
        assert(action_expected == action_actual)

def test_epsilon_greedy_policy_gives_identical_results_to_greedy_policy_when_there_is_more_than_one_greedy_maximum_and_epsilon_equals_zero():
    q_function = np.array([20, 20, 20, 20, 20, 20, 20, 20])
    random_generator_1 = default_rng(seed=0)
    random_generator_2 = copy.deepcopy(random_generator_1)
    greedy_policy = GreedyPolicy(random_generator=random_generator_1)
    epsilon_greedy_policy = GreedyPolicy(epsilon=0.0, random_generator=random_generator_2)
    for _ in range(100):
        action_expected = greedy_policy.next_action(q_function)
        action_actual = epsilon_greedy_policy.next_action(q_function)
        assert(action_expected == action_actual)

def test_epsilon_greedy_policy_gives_identical_results_to_random_policy_when_epsilon_equals_one_and_same_rng_used():
    dummy_q_function = np.array([0] * N_CHOICES)
    random_policy = create_random_policy_with_fixed_rng(N_CHOICES, 0)
    random_policy_copy = copy.deepcopy(random_policy)
    greedy_policy = GreedyPolicy(epsilon=1.0, random_policy=random_policy_copy)
    for _ in range(100):
        action_expected = random_policy.next_action()
        action_actual = greedy_policy.next_action(dummy_q_function)
        assert(action_expected == action_actual)
