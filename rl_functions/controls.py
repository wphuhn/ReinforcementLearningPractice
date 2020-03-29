"""Subroutines for updating non-class objects.

Currently, only the q function update is supported.

    Typical usage example:

    alpha = 0.5
    trajectory = [{0, 1}, {2, 3}, {4, 5}]
    rewards = [7, 13, 12]

    control = SomeControl(alpha)
    control.update(trajectory, rewards)
    q_function = control.get_q()
"""

import copy

from rl_functions.policies import GreedyPolicy

class Control(object):
    """Core functionality for control classes.
    """

    def __init__(self, epsilon, random_policy, rng, q):
        """Initializes the core control functionality.

        Args:
            q: An initial q-function to use, represented as a
                two-level nested Iterable, i.e. of form q[state][action].  If
                this argument is not supplied, an empty q-function will be
                initialized.
            epsilon: The probablity of going off-policy and selecting
                a random action from a random policy.  Behavior ranges from
                no deviation from the policy (epsilon=0.0) and complete
                randomness (epsilon=1.0).
            random_policy: The policy used to generate the random action when
                deviating from the policy.  Required when epsilon > 0.
            rng: (optional) Pseudo-random number generator (either from Python
                standard library or NumPy) used for all random number generation
                within the policy.  When None, Python's built-in PRNG will be
                used.

        Raises:
            None
        """
        if q is None:
            q = {}
        self._q = copy.deepcopy(q)
        if epsilon > 0.0:
            self._policy = GreedyPolicy(
                epsilon=epsilon,
                random_policy=random_policy,
                random_generator=rng,
            )
        else:
            self._policy = GreedyPolicy(epsilon=0.0)

    def next_action(self, state):
        """Selects an action based on the control's current policy.

        For off-policy learning, the action corresponds to the on-policy choice,
        i.e. the action the control will take (modulo epsilon-randomness)

        Args:
            state: The state for which an action should be chosen

        Returns:
            Action to take for the given state based on the current policy

        Raises:
            None
        """
        return self._policy.next_action(self._q, state)

    def get_q(self):
        """Returns the current q-function.

        Args:
            None

        Returns:
            The q function represented as a two-level nested Iterable, i.e. of
            form q[state][action].

        Raises:
            None
        """
        return copy.deepcopy(self._q)


class IterativeControl(Control):
    """Control for updating q function based on current state-action reward.

    The update method used is a simple iterative approach, with scaling factor
    alpha.  If the state exists in the q-function but the state-action pair does
    not, the state-action pair will be initialized to the maximum q value for
    that state before applying the update.  If the state does not exist in the
    q-function, the state-action pair will be initialized to a value of zero
    before applying the update.

    Attributes:
        alpha: The scaling factor for the update.
    """

    def __init__(self, alpha, epsilon=0.0, random_policy=None, rng=None, q=None):
        """Initializes the iterative control.

        Args:
            alpha: The scaling factor for the update.
            epsilon: (optional) The probablity of going off-policy and selecting
                a random action from a random policy.  Behavior ranges from
                no deviation from the policy (epsilon=0.0) and complete
                randomness (epsilon=1.0).  Default: 0.0
            random_policy: (optional) The policy used to generate the random
                action when deviating from the policy.  Required when
                epsilon > 0.
            rng: (optional) Pseudo-random number generator (either from Python
                standard library or NumPy) used for all random number generation
                within the policy.  When None, Python's built-in PRNG will be
                used.  Default: None
            q: (optional) An initial q-function to use, represented as a
                two-level nested Iterable, i.e. of form q[state][action].  If
                this argument is not supplied, an empty q-function will be
                initialized.

        Raises:
            None
        """
        if q is None:
            q = {}
        super().__init__(epsilon, random_policy, rng, q)
        self.alpha = alpha

    def update(self, trajectory, rewards):
        """Update q function based on reward from latest state-action pair.

        Args:
            trajectory: The trajectory for the episode as a list of
                (state, action) pairs, i.e. [(state0, action0),
                (state1, action1), ...].  Only the most recent element will be
                used by this algorithm.
            rewards: The rewards generated by the environment for the trajectory
                as a list, i.e. [reward0, reward1, ...].  Sister array to
                trajectory.  Only the most recent element will be used by this
                algorithm.

        Returns:
            None (updates the q-function internally).

        Raises:
            None
        """
        state, action = trajectory[-1]
        reward = rewards[-1]
        # if state does not exists in q, we insert it and the action into the
        # dictionary with a zero value
        if state not in self._q:
            self._q[state] = {}
            self._q[state][action] = 0.
        # if state exists in q but the action does not exist, we insert it into the
        # dictionary with the optimistic value
        if action not in self._q[state]:
            max_value = max(self._q[state].values())
            self._q[state][action] = max_value
        q_prev = self._q[state][action]
        q_new = q_prev + self.alpha * (reward - q_prev)
        self._q[state][action] = q_new

    def run_episode(self, env, initial_state, max_n_steps=None):
        if max_n_steps is not None and max_n_steps < 0:
            raise Exception("Episode cannot have a negative number of steps")
        # Initialize the history
        n_steps = 0
        trajectory = [(initial_state, None)]
        rewards = []
        infos = []
        state = initial_state
        while max_n_steps is None or n_steps < max_n_steps:
            # Perform the next action
            action = self.next_action(state)
            next_state, reward, done, info = env.step(action)
            # Update the history to account for the action taken and reward
            # received
            trajectory.pop()
            trajectory.append((state, action))
            rewards.append(reward)
            self.update(trajectory, rewards)
            # Move to next step
            infos.append(info)
            trajectory.append((next_state, None))
            if done:
                break
            state = next_state
            n_steps += 1
        return trajectory, rewards, infos

class OnPolicyMonteCarloControl(Control):
    """Control for updating q function based on on-policy Monte Carlo algorithm

    Currently, only first-visit Monte Carlo is currently supported.  We store
    both the q-function and number of times a state-action pair has been visited
    to make calculating the new average G value easier and reduce memory
    overhead.  If the state or state-action pair do not exist in the q-function,
    it will be added automatically.

    Attributes:
        alpha: The scaling factor for the update.
        gamma: The discount factor for rewards.
    """

    def __init__(self, gamma, epsilon=0.0, random_policy=None, rng=None, q=None, counts=None):
        """Initializes the on-policy Monte Carlo control.

        Args:
            gamma: The discount factor for rewards.
            epsilon: (optional) The probablity of going off-policy and selecting
                a random action from a random policy.  Behavior ranges from
                no deviation from the policy (epsilon=0.0) and complete
                randomness (epsilon=1.0).  Default: 0.0
            random_policy: (optional) The policy used to generate the random
                action when deviating from the policy.  Required when
                epsilon > 0.
            rng: (optional) Pseudo-random number generator (either from Python
                standard library or NumPy) used for all random number generation
                within the policy.  When None, Python's built-in PRNG will be
                used.  Default: None
            q: (optional) An initial q-function to use, represented as a
                two-level nested Iterable, i.e. of form q[state][action].  If
                this argument is not supplied, an empty q-function will be
                initialized.
            counts: (optional) The initial counts (number of times a
                state-action pair has been visited) to use, represented as a
                two-level nested Iterable, i.e. of form counts[state][action].
                If this argument is not supplied, an empty counts will be
                initialized.

        Raises:
            None
        """
        if q is None:
            q = {}
        if counts is None:
            counts = {}
        super().__init__(epsilon, random_policy, rng, q)
        self.gamma = gamma
        self._counts = copy.deepcopy(counts)

    def update(self, trajectory, rewards):
        """Update the q function using on-policy Monte Carlo control

        Args:
            trajectory: The trajectory for the episode as a list of
                (state, action) pairs, i.e. [(state0, action0),
                (state1, action1), ...].
            rewards: The rewards generated by the environment for the trajectory
                as a list, i.e. [reward0, reward1, ...].  Sister array to
                trajectory.

        Returns:
             None (updates the q-function internally).

        Raises:
            Exception when the length of rewards and trajectory is not the same.
        """
        if len(trajectory) != len(rewards):
            raise Exception(f"Trajectory and rewards have differing lengths of {len(trajectory)} and {len(rewards)}, respectively")
        traj = copy.deepcopy(trajectory)
        revs = copy.deepcopy(rewards)
        g = 0
        for time in range(len(traj)-1, -1, -1):
            state, action = traj.pop(time)
            reward = revs.pop(time)
            g = self.gamma * g + reward
            # Note: the following loop ensures that we only update q and counts
            # once for any given state-action pair based off of data that is
            # fixed (the current q/count value and the rewards).
            # If every visit is implemented, there is the possibility that partially
            # updated data will leak # downstream into the trajectoy unless we work
            # off copies of q and counts.  That being said, this leakage may
            # actually be a good thing.
            if (state, action) not in traj:
                if state not in self._q:
                    self._q[state] = {}
                    self._counts[state] = {}
                if action not in self._q[state]:
                    self._q[state][action] = 0.0
                    self._counts[state][action] = 0
                old_avg = self._q[state][action]
                old_count = self._counts[state][action]
                new_avg = (old_avg * old_count + g) / (old_count + 1)
                new_count = old_count + 1
                self._q[state][action] = new_avg
                self._counts[state][action] = new_count

    def get_counts(self):
        """Returns the current counts (# times a state-action was sampled).

        Args:
            None

        Returns:
            The counts, represented as a two-level nested Iterable, i.e. of
            form q[state][action].

        Raises:
            None
        """
        return copy.deepcopy(self._counts)


class SarsaControl(Control):
    """Control for updating q function based on the Sarsa algorithm

    Owing to the backup nature of this algorithm, depending on how one writes
    down the math for this algorithm the "current" reward could refer either to
    the reward from the previous time step (which is what enters into Sarsa) or
    the reward from the current time step.  To reduce confusion, I've decided to
    explicitly write the code in terms of the former definition.  This means
    that the rewards list should have one less element than the trajectory.

    Attributes:
        alpha: The scaling factor for the update.
        gamma: The discount factor for rewards.
    """

    def __init__(self, alpha, gamma, epsilon=0.0, random_policy=None, rng=None, q=None):
        """Initializes the Sarsa control.

        Args:
            alpha: The scaling factor for the update.
            gamma: The discount factor for rewards.
            epsilon: (optional) The probablity of going off-policy and selecting
                a random action from a random policy.  Behavior ranges from
                no deviation from the policy (epsilon=0.0) and complete
                randomness (epsilon=1.0).  Default: 0.0
            random_policy: (optional) The policy used to generate the random
                action when deviating from the policy.  Required when
                epsilon > 0.
            rng: (optional) Pseudo-random number generator (either from Python
                standard library or NumPy) used for all random number generation
                within the policy.  When None, Python's built-in PRNG will be
                used.  Default: None
            q: (optional) An initial q-function to use, represented as a
                two-level nested Iterable, i.e. of form q[state][action].  If
                this argument is not supplied, an empty q-function will be
                initialized.

        Raises:
            None
        """
        if q is None:
            q = {}
        super().__init__(epsilon, random_policy, rng, q)
        self.alpha = alpha
        self.gamma = gamma

    def update(self, trajectory, rewards):
        """Update the q function using Sarsa control

        Args:
            trajectory: The trajectory for the episode as a list of (state,
                action) pairs, i.e. [(state0, action0), (state1, action1), ...].
            rewards: The rewards generated by the environment for the trajectory
                as a list, i.e. [reward0, reward1, ...].  Sister array to
                trajectory.  Should not include the reward for the current time
                step!

        Returns:
            None (updates the q-function internally).

        Raises:
            Exception when state-action pairs are not found in the q function.
        """
        if len(rewards) == len(trajectory):
            raise Exception("Length of trajectory and rewards lists are the same; current state-action pair shouldn't yet have a reward when doing Sarsa")
        reward = rewards[-1]

        s, a = trajectory[-2]
        if s not in self._q:
            raise Exception(f"previous state {s} not found in q function")
        if a not in self._q[s]:
            raise Exception(f"previous action {a} not found in q function")

        s_p, a_p = trajectory[-1]
        if s_p not in self._q:
            raise Exception(f"current state {s_p} not found in q function")
        if a_p not in self._q[s_p]:
            raise Exception(f"current action {a_p} not found in q function")

        # Calculate new q value for old (state, action) pair
        q_prev = self._q[s][a]
        q_next = self._q[s_p][a_p]
        q_prev = q_prev + self.alpha * (reward + self.gamma * q_next - q_prev)
        # Update q and return
        self._q[s][a] = q_prev


class QLearningControl(Control):
    """Control for updating q function based on Q-learning algorithm

    Attributes:
        alpha: The scaling factor for the update.
        gamma: The discount factor for rewards.
    """

    def __init__(self, alpha, gamma, epsilon=0.0, random_policy=None, rng=None, q=None):
        """Initializes the Q-learning control.

        Args:
            alpha: The scaling factor for the update.
            gamma: The discount factor for rewards.
            epsilon: (optional) The probablity of going off-policy and selecting
                a random action from a random policy.  Behavior ranges from
                no deviation from the policy (epsilon=0.0) and complete
                randomness (epsilon=1.0).  Default: 0.0
            random_policy: (optional) The policy used to generate the random
                action when deviating from the policy.  Required when
                epsilon > 0.
            rng: (optional) Pseudo-random number generator (either from Python
                standard library or NumPy) used for all random number generation
                within the policy.  When None, Python's built-in PRNG will be
                used.  Default: None
            q: (optional) An initial q-function to use, represented as a
                two-level nested Iterable, i.e. of form q[state][action].  If
                this argument is not supplied, an empty q-function will be
                initialized.

        Raises:
            None
        """
        if q is None:
            q = {}
        super().__init__(epsilon, random_policy, rng, q)
        self.alpha = alpha
        self.gamma = gamma

    def update(self, trajectory, rewards):
        """Update the q function using the Q-learning algorithm

        Args:
            trajectory: The trajectory for the episode as a list of (state,
                action) pairs, i.e. [(state0, action0), (state1, action1), ...].
                Note that the final "pair" should consist of only a state, i.e.
                (state, )
            rewards: The rewards generated by the environment for the trajectory
                as a list, i.e. [reward0, reward1, ...].  Sister array to
                trajectory.  Should not include the reward for the current time
                step!

        Returns:
            None (updates the q-function internally).

        Raises:
            Exception when state-action pairs are not found in the q function.
        """
        if len(rewards) == len(trajectory):
            raise Exception("Length of trajectory and rewards lists are the same; current state shouldn't yet have a reward when doing Q-learning")
        if len(trajectory[-1]) > 1:
            raise Exception("current state shouldn't yet have an action when doing Q-learning")
        reward = rewards[-1]

        s, a = trajectory[-2]
        if s not in self._q:
            raise Exception(f"previous state {s} not found in q function")
        if a not in self._q[s]:
            raise Exception(f"previous action {a} not found in q function")

        s_p = trajectory[-1][0]
        if s_p not in self._q:
            raise Exception(f"current state {s_p} not found in q function")

        # Calculate new q value for old (state, action) pair
        q_prev = self._q[s][a]
        q_max = max(self._q[s_p].values())
        q_prev = q_prev + self.alpha * (reward + self.gamma * q_max - q_prev)
        # Update q and return
        self._q[s][a] = q_prev
