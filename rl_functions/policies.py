import copy
import random

def next_action_with_random_policy(env):
    return env.action_space.sample()

def next_action_with_epsilon_random_policy(env, epsilon, previous_action=None):
    if previous_action is None:
        return env.action_space.sample()
    if random.random() < epsilon:
        return env.action_space.sample()
    return copy.deepcopy(previous_action)
