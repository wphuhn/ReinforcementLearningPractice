"""Reinforcement learning policies for predicting actions of agents.

This module contains policies which generate actions based on information that
the agent has learned about the environment.  Most policies may be made
epsilon-soft by supplying an epsilon parameter, allowing the agent to
occasionally go off-policy and return a random action.  Note that when using
an epsilon-soft policy, you will need to supply a RandomPolicy during
instantiation to generate random actions.

    Typical usage:

    policy = RandomPolicy(n_actions=5)
    next_action = policy.next_action()
"""
from numpy.random import random, randint, choice

class RandomPolicy(object):
    """Policy which randomly selects between a number of actions.

    Attributes:
        n_actions: The number of actions to choose from.
    """

    def __init__(self, n_actions, random_generator=None):
        """Initializes the random policy

        Args:
            n_actions: Number of actions to choose between
            random_generator: (optional) Pseudo-random number generator
                (either from Python standard library or NumPy) used for all
                random number generation within the policy.  When None, Python's
                built-in PRNG will be used. Default: None.

        Raises:
            None
        """
        self.n_actions = n_actions
        self._random_generator = random_generator

    def next_action(self):
        """Generate the next action randomly.

        The action will be returned as an integer in the set [0, n_actions).

        Args:
            None

        Returns:
            The action to perform

        Raises:
            None
        """
        if self._random_generator is None:
            return randint(0, self.n_actions)
        return self._random_generator.integers(0, self.n_actions)

class EpsilonSoftPolicy(object):
    """Implementation of epsilon-soft functionality for other policies.

    This "policy" isn't intended to be used on its own, but rather to provide
    support for epsilon-soft variants to other policies.

    Attributes:
        No public attributes
    """

    def __init__(self, epsilon, random_policy, random_generator):
        """Initializes the epsilon-soft policy

        Args:
            epsilon: The probablity of going off-policy and selecting
                a random action from a random policy.  Behavior ranges from
                no deviation from the policy (epsilon=0.0) and complete
                randomness (epsilon=1.0).
            random_policy: The policy used to generate the random
                action when deviating from the policy.  Required when
                epsilon > 0.
            random_generator: Pseudo-random number generator
                (either from Python standard library or NumPy) used for all
                random number generation within the policy.  When None, Python's
                built-in PRNG will be used.

        Raises:
            Exception when epsilon > 0 and a random_policy has not been provided
        """
        if epsilon > 0.0 and random_policy is None:
            raise Exception("when specifying an epsilon value greater than 0, you must also provide a random policy!")
        self._epsilon = epsilon
        self._random_policy = random_policy
        self._random_generator = random_generator

    def epsilon_soft_action(self, on_policy_action):
        """Determines whether the on-policy or random action should be used.

        Args:
            on_policy_action: the action that the calling policy has decided on

        Returns:
            The action to perform

        Raises:
            None
        """
        if self._epsilon == 0.0:
            return on_policy_action

        random_action = self._random_policy.next_action()
        if self._epsilon >= 1.0:
            return random_action

        if self._random_generator is None:
            random_val = random()
        else:
            random_val = self._random_generator.random()
        if random_val < self._epsilon:
            return random_action
        return on_policy_action

class ConstantPolicy(EpsilonSoftPolicy):
    """(Epsilon-soft) constant policy which returns a constant action.

    This epsilon-soft constant policy differs from other policies in that, when
    a random action is chosen, the action for the policy will change
    permanently to the random action.  A "true" epsilon-soft variant of this
    policy can be implemented by using a DeterministicPolicy where every state
    has the same action.

    Attributes:
        action: The action to perform
    """

    def __init__(self, action, epsilon=0.0, random_policy=None, random_generator=None):
        """Initializes the (epsilon-soft) constant policy.

        Args:
            action: the on-policy action
            epsilon: (optional) The probablity of going off-policy and selecting
                a random action from a random policy.  Behavior ranges from
                no deviation from the policy (epsilon=0.0) and complete
                randomness (epsilon=1.0).  Default: 0.0
            random_policy: (optional) The policy used to generate the random
                action when deviating from the policy.  Required when
                epsilon > 0.  Default: None
            random_generator: (optional) Pseudo-random number generator
                (either from Python standard library or NumPy) used for all
                random number generation within the policy.  Default: None
                (use Python's built-in PRNG).
        """
        super().__init__(epsilon, random_policy, random_generator)
        self.action = action

    def next_action(self):
        """Generate the next action based on the (epsilon-soft) constant policy.

        There is an epsilon chance that the constant action will be changed to
        a random action chosen the the random_policy.

        Args:
            None

        Returns:
            The action to perform

        Raises:
            None
        """
        on_policy_action = self.action
        self.action = self.epsilon_soft_action(on_policy_action)
        return self.action

class GreedyPolicy(EpsilonSoftPolicy):
    """(Epsilon-soft) greedy policy to determine action based on q function.

    Attributes:
        No public attributes
    """
    def __init__(self, epsilon=0.0, random_policy=None, random_generator=None):
        """Initializes the (epsilon-soft) greedy policy.

        Args:
            epsilon: (optional) The probablity of going off-policy and selecting
                a random action from a random policy.  Behavior ranges from
                no deviation from the policy (epsilon=0.0) and complete
                randomness (epsilon=1.0).  Default: 0.0
            random_policy: (optional) The policy used to generate the random
                action when deviating from the policy.  Required when
                epsilon > 0.  Default: None
            random_generator: (optional) Pseudo-random number generator
                (either from Python standard library or NumPy) used for all
                random number generation within the policy.  Default: None
                (use Python's built-in PRNG).
        """
        super().__init__(epsilon, random_policy, random_generator)

    def greedy_action(self, q, state):
        """Generate the next action for a state by maximizing the q function.

        The action generated by this subroutine is strictly greedy.  That being
        said, it is *not* deterministic: when two or more actions maximize the q
        function for the provided state, one will be chosen at random.

        Args:
            q: The q function represented as a two-level nested Iterable, i.e.
                of form q[state][action].
            state: the state whose associated action is being requested

        Returns:
            The greedy action to perform for the given state

        Raises:
            None
        """
        q_state = q[state]
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

    def next_action(self, q, state):
        """Generate the next action based on the (epsilon-soft) greedy policy.

        There is an epsilon chance that a random action will be taken instead,
        as decided by the associated random_policy.

        Args:
            q: The q function represented as a two-level nested Iterable, i.e.
                of form q[state][action].
            state: the state whose associated action is being requested

        Returns:
            The action to perform for the given state

        Raises:
            None
        """
        on_policy_action = self.greedy_action(q, state)
        action = self.epsilon_soft_action(on_policy_action)
        return action

class DeterministicPolicy(EpsilonSoftPolicy):
    """Deterministic policy which outputs a unique action given a state.

    By default, the policy is empty.  State-action pairs may be registered to
    the policy directly using add_transitions() or may be infered from a
    q function using generate_greedily_from_q.

    Attributes:
        No public attributes
    """
    def __init__(self, epsilon=0.0, random_policy=None, random_generator=None):
        """Inits deterministic policy with no registered state-action pairs."""
        super().__init__(epsilon, random_policy, random_generator)
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
        on_policy_action = self._transitions[state]
        action = self.epsilon_soft_action(on_policy_action)
        return action

    def generate_greedily_from_q(self, q):
        """Adds state-action pairs to the based on greedy q-function maximizing.

        If the q function has non-unique maximum values for any state, this
        method for initializing the deterministic policy cannot be used.

        Args:
            q: The q function represented as a two-level nested Iterable, i.e.
                of form q[state][action].

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
