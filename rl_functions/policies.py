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

class ConstantPolicy:
    def __init__(self, action, epsilon=0.0, random_policy=None, random_generator=None):
        self.action = action
        if epsilon > 0.0 and random_policy is None:
            raise Exception("when specifying an epsilon value greater than 0, you must also provide a random policy!")
        self._epsilon = epsilon
        self._random_policy = random_policy
        self._random_generator = random_generator

    def next_action(self):
        if self._epsilon == 0.0:
            return self.action

        random_action = self._random_policy.next_action()
        if self._epsilon >= 1.0:
            return random_action

        if self._random_generator is None:
            random_val = random()
        else:
            random_val = self._random_generator.random()
        if self.action is None or random_val < self._epsilon:
            self.action = random_action
        return self.action

def next_action_with_random_policy(num_choices, random_generator=None):
    if random_generator is None:
        return randint(0, num_choices)
    else:
        return random_generator.integers(0, num_choices)

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
