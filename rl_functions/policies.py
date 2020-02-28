import numpy as np
from numpy.random import random, randint, choice

class RandomPolicy(object):
    def __init__(self, num_choices, random_generator=None):
        self._num_choices = num_choices
        self._random_generator = random_generator

    def next_action(self):
        if self._random_generator is None:
            return randint(0, self._num_choices)
        return self._random_generator.integers(0, self._num_choices)

class ConstantPolicy(object):
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
        if random_val < self._epsilon:
            self.action = random_action
        return self.action

class GreedyPolicy(object):
    def __init__(self, epsilon=0.0, random_policy=None, random_generator=None):
        if epsilon > 0.0 and random_policy is None:
            raise Exception("when specifying an epsilon value greater than 0, you must also provide a random policy!")
        self._epsilon = epsilon
        self._random_policy = random_policy
        self._random_generator = random_generator

    def _greedy_action(self, q_function):
        max_indices = np.where(q_function == q_function.max())[0]
        if len(max_indices) == 1:
            return max_indices[0]
        if self._random_generator is None:
            return choice(max_indices)
        return self._random_generator.choice(max_indices)

    def next_action(self, q_function):
        greedy_action = self._greedy_action(q_function)
        if self._epsilon == 0.0:
            return greedy_action

        random_action = self._random_policy.next_action()
        if self._epsilon >= 1.0:
            return random_action

        if self._random_generator is None:
            random_val = random()
        else:
            random_val = self._random_generator.random()
        if random_val < self._epsilon:
            return random_action
        return greedy_action
