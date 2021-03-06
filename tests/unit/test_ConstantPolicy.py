import copy

from numpy.random import default_rng
import pytest

from rl_functions.policies import ConstantPolicy

from constants import N_ACTIONS
from utilities import create_rngs_with_fixed_seed

def test_constant_policy_gives_a_consistant_answer_when_it_is_run_repeatedly():
    expected = 3
    constant_policy = ConstantPolicy(expected)
    for _ in range(100):
        actual = constant_policy.next_action()
        assert expected == actual

def test_epsilon_constant_policy_outcome_doesnt_change_when_epsilon_is_zero():
    expected = 3
    constant_policy = ConstantPolicy(expected, epsilon=0.0)
    for _ in range(100):
        actual = constant_policy.next_action()
        assert expected == actual

def test_creating_constant_policy_throws_exception_when_epsilon_is_greater_than_zero_and_random_policy_not_provided():
    dummy = 3
    with pytest.raises(Exception) as excinfo:
        _ = ConstantPolicy(dummy, epsilon=1.0)
    assert "when specifying an epsilon value greater than 0, you must also provide a random policy!" in str(excinfo.value)

def test_epsilon_constant_policy_gives_identical_results_to_random_policy_when_epsilon_equals_one_and_same_rng_used():
    # Not the best test in the world, I know, but it's an important edge case
    # We generate an action from a random policy
    random_policy, _ = create_rngs_with_fixed_seed(N_ACTIONS, 0, 0)
    expected = random_policy.next_action()
    # Then we copy the state of the random policy *after* action has been taken
    random_policy_copy = copy.deepcopy(random_policy)
    # We use the action and random policy copy to create a new epsilon-constant
    # policy, which *should* be just the same random policy based on the way
    # we've set it up
    constant_policy = ConstantPolicy(
        expected,
        epsilon=1.0,
        random_policy=random_policy_copy,
    )
    # And the two should give identical results
    for _ in range(100):
        expected = random_policy.next_action()
        actual = constant_policy.next_action()
        assert expected == actual

def test_epsilon_constant_policy_gives_deterministic_results_when_an_rng_with_a_fixed_seed_is_supplied():
    initial_action = 3
    policy, epsilon_rng = create_rngs_with_fixed_seed(N_ACTIONS, 0, 0)
    constant_policy = ConstantPolicy(
        initial_action,
        epsilon=0.5,
        random_policy=policy,
        random_generator=epsilon_rng,
    )
    actions_expected = [3, 5, 4, 2, 2]
    for expected in actions_expected:
        actual = constant_policy.next_action()
        assert expected == actual

def test_two_epsilon_constant_policies_give_the_same_results_when_one_is_a_copy_of_the_other():
    initial_action = 3
    policy_1, epsilon_rng_1 = create_rngs_with_fixed_seed(N_ACTIONS, 0, 0)
    policy_2 = copy.deepcopy(policy_1)
    epsilon_rng_2 = copy.deepcopy(epsilon_rng_1)
    constant_policy_1 = ConstantPolicy(
        initial_action,
        epsilon=0.5,
        random_policy=policy_1,
        random_generator=epsilon_rng_1,
    )
    constant_policy_2 = ConstantPolicy(
        initial_action,
        epsilon=0.5,
        random_policy=policy_2,
        random_generator=epsilon_rng_2,
    )
    for _ in range(100):
        expected = constant_policy_1.next_action()
        actual = constant_policy_2.next_action()
        assert expected == actual
