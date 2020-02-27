import copy

from numpy.random import default_rng
import pytest

from rl_functions.policies import ConstantPolicy

from constants import N_CHOICES
from utilities import create_random_policy_with_fixed_rng

def test_constant_policy_outcome_doesnt_change():
    action_expected = 3
    constant_policy = ConstantPolicy(action_expected)
    for _ in range(100):
        action_actual = constant_policy.next_action()
        assert(action_expected == action_actual)

def test_epsilon_constant_policy_outcome_doesnt_change_when_epsilon_is_zero():
    action_expected = 3
    constant_policy = ConstantPolicy(action_expected, epsilon=0.0)
    for _ in range(100):
        action_actual = constant_policy.next_action()
        assert(action_expected == action_actual)

def test_creating_constant_policy_throws_exception_when_epsilon_is_greater_than_zero_and_random_policy_not_provided():
    dummy_action = 3
    with pytest.raises(Exception) as excinfo:
        _ = ConstantPolicy(dummy_action, epsilon=1.0)
    assert "when specifying an epsilon value greater than 0, you must also provide a random policy!" in str(excinfo.value)

def test_epsilon_constant_policy_gives_identical_results_to_random_policy_when_epsilon_equals_one_and_same_rng_used():
    # Not the best test in the world, I know, but it's an important edge case
    # We generate an action from a random policy
    random_policy = create_random_policy_with_fixed_rng(N_CHOICES, 0)
    action_expected = random_policy.next_action()
    # Then we copy the state of the random policy *after* the action has been taken
    random_policy_copy = copy.deepcopy(random_policy)
    # We use the action and random policy copy to create a new epsilon-constant policy, which *should* be just
    # the same random policy based on the way we've set it up
    constant_policy = ConstantPolicy(action_expected, epsilon=1.0, random_policy=random_policy_copy)
    # And the two should give identical results
    for _ in range(100):
        action_expected = random_policy.next_action()
        action_actual = constant_policy.next_action()
        assert(action_expected == action_actual)

def test_seeded_epsilon_constant_policy_gives_expected_results():
    initial_action = 3
    random_policy = create_random_policy_with_fixed_rng(N_CHOICES, 0)
    random_generator = default_rng(seed=0)
    constant_policy = ConstantPolicy(initial_action, epsilon=0.5, random_policy=random_policy, random_generator=random_generator)
    actions_expected = [3, 5, 4, 2, 2]
    for action_expected in actions_expected:
        action_actual = constant_policy.next_action()
        assert(action_expected == action_actual)

def test_two_epsilon_constant_policies_with_random_generators_in_parallel_give_the_same_results():
    initial_action = 3
    random_policy_1 = create_random_policy_with_fixed_rng(N_CHOICES, 0)
    random_policy_2 = copy.deepcopy(random_policy_1)
    random_generator_1 = default_rng(seed=0)
    random_generator_2 = copy.deepcopy(random_generator_1)
    constant_policy_1 = ConstantPolicy(initial_action, epsilon=0.5, random_policy=random_policy_1, random_generator=random_generator_1)
    constant_policy_2 = ConstantPolicy(initial_action, epsilon=0.5, random_policy=random_policy_2, random_generator=random_generator_2)
    for _ in range(100):
        action_1 = constant_policy_1.next_action()
        action_2 = constant_policy_2.next_action()
        assert(action_1 == action_2)
