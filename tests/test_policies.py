import copy
import random

import numpy as np
from numpy.random import default_rng
import pytest

from rl_functions.policies import next_action_with_epsilon_random_policy, next_action_with_greedy_policy, \
    next_action_with_epsilon_greedy_policy

from constants import ENV_NAME, N_CHOICES
from utilities import create_random_policy_with_fixed_rng

@pytest.fixture
def random_policy():
    return create_random_policy_with_fixed_rng(N_CHOICES, 0)

def test_seeded_random_policy_gives_expected_results(random_policy):
    actions_expected = [6, 5, 4, 2, 2]
    random_generator = default_rng(seed=0)
    for action_expected in actions_expected:
        action_actual = random_policy.next_action()
        assert(action_expected == action_actual)

def test_two_identical_random_policies_in_parallel_give_the_same_results(random_policy):
    random_policy_copy = copy.deepcopy(random_policy)
    for _ in range(100):
        action_orig = random_policy.next_action()
        action_copy = random_policy_copy.next_action()
        assert(action_orig == action_copy)

def test_seeded_epsilon_random_policy_gives_expected_results():
    actions_expected = [6, 5, 5, 5, 5]
    random_generator = default_rng(seed=0)
    action_actual = None
    for action_expected in actions_expected:
        action_actual = next_action_with_epsilon_random_policy(N_CHOICES, 0.5, action_actual, random_generator=random_generator)
        assert(action_expected == action_actual)

def test_two_epsilon_random_policies_with_random_generators_in_parallel_give_the_same_results():
    action_1 = None
    action_2 = None
    random_generator_1 = default_rng(seed=0)
    random_generator_2 = default_rng(seed=0)
    for _ in range(100):
        action_1 = next_action_with_epsilon_random_policy(N_CHOICES, 0.5, action_1, random_generator=random_generator_1)
        action_2 = next_action_with_epsilon_random_policy(N_CHOICES, 0.5, action_2, random_generator=random_generator_2)
        assert(action_1 == action_2)

def test_epsilon_random_policy_outcome_doesnt_change_when_epsilon_equals_zero():
    # Let it run once to pick the first step
    random_generator = default_rng(seed=0)
    action_expected = next_action_with_epsilon_random_policy(N_CHOICES, 0.0, None, random_generator=random_generator)
    action_actual = copy.deepcopy(action_expected)
    for _ in range(100):
        action_actual = next_action_with_epsilon_random_policy(N_CHOICES, 0.0, action_actual, random_generator=random_generator)
        assert(action_expected == action_actual)

def test_random_policy_gives_identical_results_to_epsilon_random_policy_when_epsilon_equals_one(random_policy):
    random_generator_2 = default_rng(seed=0)
    action_actual = None
    for _ in range(100):
        action_expected = random_policy.next_action()
        action_actual = next_action_with_epsilon_random_policy(N_CHOICES, 1.0, action_actual, random_generator=random_generator_2)
        assert(action_expected == action_actual)

def test_greedy_policy_gives_expected_results():
    q_function = np.array([17.1, -5.5, 20.2, -20.9, 5.1, 1.0, -1.0, 1.5])
    action_expected = 2
    for _ in range(100):
        action_actual = next_action_with_greedy_policy(q_function)
        assert(action_expected == action_actual)

def test_greedy_policy_breaks_ties_in_a_pseudorandom_fashion():
    q_function = np.array([20, 20, 20, 20, 20, 20, 20, 20])
    actions_expected = [6, 5, 4, 2, 2]
    random_generator = default_rng(seed=0)
    for action_expected in actions_expected:
        action_actual = next_action_with_greedy_policy(q_function, random_generator=random_generator)
        assert(action_expected == action_actual)

def test_epsilon_greedy_policy_gives_expected_results():
    q_function = np.array([17.1, -5.5, 20.2, -20.9, 5.1, 1.0, -1.0, 1.5])
    actions_expected = [6, 5, 2, 2, 2]
    random_generator = default_rng(seed=0)
    for action_expected in actions_expected:
        action_actual = next_action_with_epsilon_greedy_policy(q_function, 0.5, N_CHOICES, random_generator=random_generator)
        assert(action_expected == action_actual)

def test_epsilon_greedy_policy_gives_identical_results_to_greedy_policy_when_epsilon_equals_zero():
    q_function = np.array([17.1, -5.5, 20.2, -20.9, 5.1, 1.0, -1.0, 1.5])
    random_generator_1 = default_rng(seed=0)
    random_generator_2 = default_rng(seed=0)
    for _ in range(100):
        action_expected = next_action_with_greedy_policy(q_function, random_generator=random_generator_1)
        action_actual = next_action_with_epsilon_greedy_policy(q_function, 0.0, N_CHOICES, random_generator=random_generator_2)
        assert(action_expected == action_actual)

def test_epsilon_greedy_policy_gives_identical_results_to_random_policy_when_epsilon_equals_one(random_policy):
    q_function = np.array([17.1, -5.5, 20.2, -20.9, 5.1, 1.0, -1.0, 1.5])
    random_generator_2 = default_rng(seed=0)
    for _ in range(100):
        action_expected = random_policy.next_action()
        action_actual = next_action_with_epsilon_greedy_policy(q_function, 1.0, N_CHOICES, random_generator=random_generator_2)
        assert(action_expected == action_actual)
