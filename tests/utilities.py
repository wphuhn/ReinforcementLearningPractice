from numpy.random import default_rng

from rl_functions.policies import RandomPolicy

def create_rngs_with_fixed_seed(n_actions, policy_seed, epsilon_seed):
    policy_rng = default_rng(seed=policy_seed)
    epsilon_rng = default_rng(seed=epsilon_seed)
    policy = RandomPolicy(n_actions, random_generator=policy_rng)
    return policy, epsilon_rng
