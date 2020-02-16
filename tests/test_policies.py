import copy
import random
from numpy.random import default_rng

from rl_functions.policies import next_action_with_random_policy, \
    next_action_with_epsilon_random_policy
from rl_functions.utilities import make_envs

from constants import ENV_NAME

def test_seeded_random_policy_gives_expected_results():
   env, _ = make_envs(ENV_NAME, env_seed=0, env_action_seed=0)
   actions_expected = [5, 0, 3, 3, 7]
   for action_expected in actions_expected:
       action_actual = next_action_with_random_policy(env)
       assert(action_expected == action_actual)

def test_two_random_policies_in_parallel_give_the_same_results():
   env_1, _ = make_envs(ENV_NAME, env_seed=0, env_action_seed=0)
   env_2, _ = make_envs(ENV_NAME, env_seed=0, env_action_seed=0)
   for _ in range(5):
      action_1 = next_action_with_random_policy(env_1)
      action_2 = next_action_with_random_policy(env_2)
      assert(action_1 == action_2)

def test_seeded_epsilon_random_policy_gives_expected_results():
   env, _ = make_envs(ENV_NAME, env_seed=0, env_action_seed=0)
   actions_expected = [5, 0, 3, 3, 3]
   random_generator = default_rng(seed=0)
   action_actual = None
   for action_expected in actions_expected:
       action_actual = next_action_with_epsilon_random_policy(env, 0.5, action_actual, random_generator=random_generator)
       assert(action_expected == action_actual)

def test_two_epsilon_random_policies_with_random_generators_in_parallel_give_the_same_results():
   env_1, _ = make_envs(ENV_NAME, env_seed=0, env_action_seed=0)
   env_2, _ = make_envs(ENV_NAME, env_seed=0, env_action_seed=0)
   action_1 = None
   action_2 = None
   random_generator_1 = default_rng(seed=0)
   random_generator_2 = default_rng(seed=0)
   for _ in range(5):
      action_1 = next_action_with_epsilon_random_policy(env_1, 0.5, action_1, random_generator=random_generator_1)
      action_2 = next_action_with_epsilon_random_policy(env_2, 0.5, action_2, random_generator=random_generator_2)
      assert(action_1 == action_2)

def test_epsilon_random_policy_outcome_doesnt_changes_when_epsilon_equals_zero():
   env, _ = make_envs(ENV_NAME, env_seed=0, env_action_seed=0)
   # Let it run once to pick the first step
   action_expected = next_action_with_epsilon_random_policy(env, 0.0, None)
   action_actual = copy.deepcopy(action_expected)
   for _ in range(5):
       action_actual = next_action_with_epsilon_random_policy(env, 0.0, action_actual)
       assert(action_expected == action_actual)

def test_random_policy_gives_identical_results_to_epsilon_random_policy_when_epsilon_equals_one():
   env_random, _ = make_envs(ENV_NAME, env_seed=0, env_action_seed=0)
   env_epsilon, _ = make_envs(ENV_NAME, env_seed=0, env_action_seed=0)
   action_actual = None
   for _ in range(5):
       action_expected = next_action_with_random_policy(env_random)
       action_actual = next_action_with_epsilon_random_policy(env_epsilon, 1.0, action_actual)
       assert(action_expected == action_actual)
