"""Reinforcement learning policies for predicting actions of agents.

This module contains policies which generate actions based on information that
the agent has learned about the environment.  Most policies may be made
epsilon-soft by supplying an epsilon parameter, allowing the agent to
occasionally go off-policy and return a random action.  Note that when using
an epsilon-soft policy, you will need to supply a RandomPolicy during
instantiation to generate random actions.

    Typical usage:

    policy = RandomPolicy(num_actions=5)
    next_action = policy.next_action()
"""
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
    """Deterministic policy which outputs a unique action given a state.

    By default, the policy is empty.  State-action pairs may be registered to
    the policy directly using add_transitions() or may be infered from a
    q function using generate_greedily_from_q.

    Attributes:
        None
    """
    def __init__(self):
        """Inits deterministic policy with no registered state-action pairs."""
        self._transitions = {}

    def add_transitions(self, trans_dict):
        """Add state-action pairs to the determinstic policy.

        Args:
            trans_dict: a dictionary encoding the pairs to add, with the states
                as keys and the actions as the associate values, i.e. of form
                {state: action}

        Returns:
            None

        Raises:
            Exception when state or action index is negative
        """
        for state, action in trans_dict.items():
            if state < 0:
                raise Exception("State index in deterministic policy must be non-negative")
            if action < 0:
                raise Exception("Action index in deterministic policy must be non-negative")
        self._transitions.update(trans_dict)

    def next_action(self, state):
        """Generate the next action predicted by the policy for a state.

        Args:
            state: the state whose associated action is being requested

        Returns:
            The action to perform for the given state

        Raises:
            Exception when the state supplied by the user has no action
            associated with it
        """
        if state not in self._transitions:
            raise Exception(f"State {state} in deterministic policy has no action registered with it")
        return self._transitions[state]

    def generate_greedily_from_q(self, q):
        """Adds state-action pairs to the based on greedy q-function maximizing.

        If the q function has non-unique maximum values for any state, this
        method for initializing the deterministic policy cannot be used.

        q: The q function represented as a two-level nested Iterable, i.e. of
            form q[state][action].

        Returns:
            None

        Raises:
            Exception when a state has more than one maximizing action in the
            q function.
        """
        for state in q:
            q_state = q[state]
            max_value = max(q_state.values())
            action = [
                action for action, value in q_state.items() if (value == max_value)
            ]
            if len(action) > 1:
                raise Exception(f"state {state} does not have a unique greedy action in q function, cannot generate a deterministic policy for it")
            self.add_transitions({state: action[0]})
