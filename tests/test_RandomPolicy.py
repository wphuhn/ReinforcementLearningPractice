import copy

from numpy.random import default_rng
import pytest

from constants import N_CHOICES
from utilities import create_random_policy_with_fixed_rng

@pytest.fixture
def random_policy():
    return create_random_policy_with_fixed_rng(N_CHOICES, 0)

def test_seeded_random_policy_gives_expected_results(random_policy):
    random_generator = default_rng(seed=0)
    actions_expected = [6, 5, 4, 2, 2]
    for action_expected in actions_expected:
        action_actual = random_policy.next_action()
        assert(action_expected == action_actual)

def test_two_identical_random_policies_in_parallel_give_the_same_results(random_policy):
    random_policy_copy = copy.deepcopy(random_policy)
    for _ in range(100):
        action_orig = random_policy.next_action()
        action_copy = random_policy_copy.next_action()
        assert(action_orig == action_copy)
