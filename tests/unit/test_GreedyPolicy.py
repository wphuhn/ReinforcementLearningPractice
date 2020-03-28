import copy

import numpy as np
from numpy.random import default_rng
import pytest

from rl_functions.policies import GreedyPolicy

from utilities import create_rngs_with_fixed_seed

def test_creating_greedy_policy_throws_exception_when_epsilon_is_greater_than_zero_and_random_policy_not_provided():
    with pytest.raises(Exception) as excinfo:
        _ = GreedyPolicy(epsilon=1.0)
    assert "when specifying an epsilon value greater than 0, you must also provide a random policy!" in str(excinfo.value)

def test_greedy_policy_picks_unique_greedy_answer_when_there_is_a_clear_greedy_maximum():
    state = 5
    expected = 2
    q_function = {
        state: {
            0: 17.1,
            1: -5.5,
            expected: 20.2,
            3: -20.9,
            4: 5.1,
            5: 1.0,
            6: -1.0,
            7: 1.5,
        }
    }
    greedy_policy = GreedyPolicy()
    for _ in range(100):
        actual = greedy_policy.next_action(q_function, state)
        assert expected == actual

def test_epsilon_greedy_policy_picks_unique_greedy_answer_when_there_is_a_unique_greedy_maximum_and_epsilon_is_zero():
    state = 5
    expected = 2
    q_function = {
        state: {
            0: 17.1,
            1: -5.5,
            expected: 20.2,
            3: -20.9,
            4: 5.1,
            5: 1.0,
            6: -1.0,
            7: 1.5,
        }
    }
    greedy_policy = GreedyPolicy(epsilon=0.0)
    for _ in range(100):
        actual = greedy_policy.next_action(q_function, state)
        assert expected == actual

def test_greedy_policy_breaks_ties_in_a_pseudorandom_fashion_when_there_is_more_than_one_greedy_maximum():
    state = 5
    q_function = {
        state: {
            0: 20,
            1: 20,
            2: 20,
            3: 20,
            4: 20,
            5: 20,
            6: 20,
            7: 20,
        }
    }
    random_generator = default_rng(seed=0)
    greedy_policy = GreedyPolicy(random_generator=random_generator)
    actions_expected = [6, 5, 4, 2, 2]
    for expected in actions_expected:
        actual = greedy_policy.next_action(q_function, state)
        assert expected == actual

def test_epsilon_greedy_policy_gives_deterministic_results_when_an_rng_with_a_fixed_seed_is_supplied():
    state = 5
    q_function = {
        state: {
            0: 17.1,
            1: -5.5,
            2: 20.2,
            3: -20.9,
            4: 5.1,
            5: 1.0,
            6: -1.0,
            7: 1.5,
        }
    }
    policy, rng = create_rngs_with_fixed_seed(8, 0, 0)
    greedy_policy = GreedyPolicy(
        epsilon=0.5,
        random_policy=policy,
        random_generator=rng,
    )
    actions_expected = [2, 5, 4, 2, 2, 2]
    for expected in actions_expected:
        actual = greedy_policy.next_action(q_function, state)
        assert expected == actual

def test_epsilon_greedy_policy_gives_identical_results_to_greedy_policy_when_there_is_a_unique_greedy_maximum_and_epsilon_equals_zero():
    state = 5
    q_function = {
        state: {
            0: 17.1,
            1: -5.5,
            2: 20.2,
            3: -20.9,
            4: 5.1,
            5: 1.0,
            6: -1.0,
            7: 1.5,
        }
    }
    greedy_policy = GreedyPolicy()
    epsilon_greedy_policy = GreedyPolicy(epsilon=0.0)
    for _ in range(100):
        expected = greedy_policy.next_action(q_function, state)
        actual = epsilon_greedy_policy.next_action(q_function, state)
        assert expected == actual

def test_epsilon_greedy_policy_gives_identical_results_to_greedy_policy_when_there_is_more_than_one_greedy_maximum_and_epsilon_equals_zero():
    state = 5
    q_function = {
        state: {
            0: 20,
            1: 20,
            2: 20,
            3: 20,
            4: 20,
            5: 20,
            6: 20,
            7: 20,
        }
    }
    random_generator_1 = default_rng(seed=0)
    random_generator_2 = copy.deepcopy(random_generator_1)
    greedy_policy = GreedyPolicy(random_generator=random_generator_1)
    epsilon_greedy_policy = GreedyPolicy(
        epsilon=0.0,
        random_generator=random_generator_2,
    )
    for _ in range(100):
        expected = greedy_policy.next_action(q_function, state)
        actual = epsilon_greedy_policy.next_action(q_function, state)
        assert expected == actual

def test_epsilon_greedy_policy_gives_identical_results_to_random_policy_when_epsilon_equals_one_and_same_rng_used():
    state = 5
    dummy_q_function = {
        state: {
            0: 17.1,
            1: -5.5,
            2: 20.2,
            3: -20.9,
            4: 5.1,
            5: 1.0,
            6: -1.0,
            7: 1.5,
        }
    }
    random_policy, _ = create_rngs_with_fixed_seed(8, 0, 0)
    random_policy_copy = copy.deepcopy(random_policy)
    greedy_policy = GreedyPolicy(epsilon=1.0, random_policy=random_policy_copy)
    for _ in range(100):
        expected = random_policy.next_action()
        actual = greedy_policy.next_action(dummy_q_function, state)
        assert expected == actual
