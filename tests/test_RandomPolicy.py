import copy

import pytest

from constants import N_ACTIONS
from utilities import create_random_policy_with_fixed_rng

@pytest.fixture
def random_policy():
    return create_random_policy_with_fixed_rng(N_ACTIONS, 0)

def test_random_policy_gives_deterministic_results_when_an_rng_with_a_fixed_seed_is_supplied(random_policy):
    actions_expected = [6, 5, 4, 2, 2]
    for expected in actions_expected:
        actual = random_policy.next_action()
        assert expected == actual

def test_two_random_policies_give_the_same_results_when_one_is_a_copy_of_the_other(random_policy):
    random_policy_copy = copy.deepcopy(random_policy)
    for _ in range(100):
        expected = random_policy.next_action()
        actual = random_policy_copy.next_action()
        assert expected == actual
