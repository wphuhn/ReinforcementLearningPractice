import copy

import numpy as np
from numpy.random import random, randint, choice

class RandomPolicy:
    def __init__(self, num_choices, random_generator=None):
        self._num_choices = num_choices
        self._random_generator = random_generator

    def next_action(self):
        if self._random_generator is None:
            return randint(0, self._num_choices)
        else:
            return self._random_generator.integers(0, self._num_choices)

def next_action_with_random_policy(num_choices, random_generator=None):
    if random_generator is None:
        return randint(0, num_choices)
    else:
        return random_generator.integers(0, num_choices)

def next_action_with_epsilon_random_policy(num_choices, epsilon, previous_action=None, random_generator=None):
    random_action = next_action_with_random_policy(num_choices, random_generator=random_generator)
    if epsilon >= 1.0:
        return random_action

    if random_generator is None:
        random_val = random()
    else:
        random_val = random_generator.random()
    if previous_action is None or random_val < epsilon:
        return random_action
    return copy.deepcopy(previous_action)

def next_action_with_greedy_policy(q_function, random_generator=None):
    max_indices = np.where(q_function == q_function.max())[0]
    if len(max_indices) == 1:
        return max_indices[0]
    if random_generator is None:
        return choice(max_indices)
    else:
        return random_generator.choice(max_indices)

def next_action_with_epsilon_greedy_policy(q_function, epsilon, num_choices, random_generator=None):
    random_action = next_action_with_random_policy(num_choices, random_generator=random_generator)
    if epsilon >= 1.0:
        return random_action

    if random_generator is None:
        random_val = random()
    else:
        random_val = random_generator.random()
    if random_val < epsilon:
        return random_action

    return next_action_with_greedy_policy(q_function, random_generator=random_generator)
