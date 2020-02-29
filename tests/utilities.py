from numpy.random import default_rng

from rl_functions.policies import RandomPolicy

def create_random_policy_with_fixed_rng(n_actions, seed):
    random_generator = default_rng(seed=seed)
    return RandomPolicy(n_actions, random_generator=random_generator)
