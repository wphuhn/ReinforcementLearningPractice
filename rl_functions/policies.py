import numpy as np
from numpy.random import random, randint, choice

class RandomPolicy(object):
    def __init__(self, num_actions, random_generator=None):
        self._num_actions = num_actions
        self._random_generator = random_generator

    def next_action(self):
        if self._random_generator is None:
            return randint(0, self._num_actions)
        return self._random_generator.integers(0, self._num_actions)

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

    def _greedy_action(self, q_function, state):
        q_state = q_function[state]
        # TODO: The following two lines should be better optimized
        max_value = max(q_state.values())
        max_actions = [
            action for action, value in q_state.items() if (value == max_value)
        ]
        if len(max_actions) == 1:
            return max_actions[0]
        if self._random_generator is None:
            return choice(max_actions)
        return self._random_generator.choice(max_actions)

    def next_action(self, q_function, state):
        greedy_action = self._greedy_action(q_function, state)
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

class DeterministicPolicy(object):
    def __init__(self, n_states, n_actions):
        self._n_states = n_states
        self._n_actions = n_actions
        self._transitions = {}

    def add_transitions(self, trans_dict):
        for state, action in trans_dict.items():
            if state < 0:
                raise Exception("State index in deterministic policy must be non-negative")
            if state >= self._n_states:
                raise Exception(f"State index {state} in deterministic policy is larger than the maximum state index of {self._n_states-1}")
            if action < 0:
                raise Exception("Action index in deterministic policy must be non-negative")
            if action >= self._n_actions:
                raise Exception(f"Action index {action} in deterministic policy is larger than the maximum action index of {self._n_actions-1}")
        self._transitions.update(trans_dict)

    def next_action(self, state):
        if state not in self._transitions:
            raise Exception(f"State {state} in deterministic policy has no action registered with it")
        return self._transitions[state]
