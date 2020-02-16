import copy

from numpy.random import random

def next_action_with_random_policy(env):
    return env.action_space.sample()

def next_action_with_epsilon_random_policy(env, epsilon, previous_action=None, random_generator=None):
    next_action = env.action_space.sample()
    if random_generator is None:
        random_val = random()
    else:
        random_val = random_generator.random()

    if previous_action is None or random_val < epsilon:
        return next_action
    return copy.deepcopy(previous_action)
