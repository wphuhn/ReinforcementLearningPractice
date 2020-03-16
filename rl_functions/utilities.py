"""Miscellaneous helper subroutines that haven't found a place elsewhere.
"""
from math import floor
from time import time

import gym
from gym import wrappers
import numpy as np

def run_summary(elapsed_time, episode, n_steps, cumul_reward):
    """Return a string summarizing the current episode.

    Args:
        elapsed_time: Total time (in seconds).
        episode: Current episode number.
        n_steps: Total number of steps taken during this episode.
        cumul_reward: Cumulative reward for this episode.

    Returns:
        A string with a summary of the current episode.

    Raises:
        None
    """
    return f"Episode: {episode} , Elapsed Time for Episode: {elapsed_time} s , Num. Steps in Episode: {n_steps} , Cumulative Episode Reward: {cumul_reward}"

def step_statistics(timestep, reward, cumul_reward, info):
    """Return a string summarizing the current time step in an episode.

    Args:
        timestep: Current timestep number in episode.
        reward: Reward for current timestep.
        cumul_reward: Cumulative reward for episode.
        info: Miscellaneous info to be output.

    Returns:
        A string with the statistics of the current timestep.

    Raises:
        None
    """
    return f"    Step: {timestep} , Reward This Step: {reward} , Cumulative Reward This Episode: {cumul_reward} , Info: {info}"

def make_envs(env_name, output_movie=False, output_folder=None,
              env_state_seed=None, env_action_seed=None,
              max_steps_per_episode=None):
    """Factory method to create OpenAI environments

    Args:
        env_name: The OpenAI environment to load
        output_movie: (optional) Whether to render a movie
        output_folder: (optional) If rendering a movie, the folder to place
            movies in
        env_seed: (optional) The pseudorandom seed to initialize the state space
            of the environment
        env_action_speed: (optional) The pseudorandom seed to initialize the
            action space of the environment
        max_steps_per_episode: (optional) The maximum number of steps that may
            be taken per episode before the environment transitions into the
            terminal state

    Returns:
        A tuple with three elements:  the active OpenAI environment, the
        OpenAI environment before being wrapped by a monitor (only relevant
        when rending a movie), and the initial state of the environment)

    Raises:
        None
    """
    # Create the environments
    if output_movie:
        env_raw = gym.make(env_name)
        if output_folder is None:
            output_folder = "./videos/" + str(time()) + "/"
        env = wrappers.Monitor(env_raw, output_folder, force=True)
    else:
        env_raw = None
        env = gym.make(env_name)
    # Set the seeds for the PRNG of the environment
    if env_state_seed is not None:
        env.seed(env_state_seed)
    if env_action_seed is not None:
        env.action_space.np_random.seed(env_action_seed)
    # Initialize the environment (does not change seed) and record the inital
    # state
    observation = env.reset()
    # Modify the number of steps per epsiode, which can be set painfully low
    # for some environments
    # Yes, I'm modifying a protected variable here, but it's what OpenAI
    # developers recommended on their GitHub, so it's on their heads, not mine.
    if max_steps_per_episode is not None:
        env._max_episode_steps = max_steps_per_episode
        if output_movie:
            env_raw._max_episode_steps = max_steps_per_episode
    return env, env_raw, observation

def close_envs(env, env_raw=None):
    """Close the OpenAI environments used

    Args:
        env: the current environment
        env_raw: (optional) the environment before being wrapped by a Monitor
            (only needed when rendering a movie)

    Returns:
        None

    Raises:
        None
    """
    env.close()
    if env_raw is not None:
        env_raw.close()

class StateEncoder(object):
    """Transforms a point in n-dimensional continuous space to a discrete value.

    This transformation is performed via a standard grid-based approach, where
    a grid is imposed over the continuous space and all points within a grid
    element are mapped to the same discrete value.  Any coorindate for a point
    which lies outside of the grid is "clamped" to the min/max value for the
    grid.

    The discrete value is generated via a standard mixed-radix expansion, where
    the number of grid elements is the base.

    Attributes:
        No public attributes

    Usage:
        grid = [2, 2]
        min_values = [0, 1]
        max_values = [0, 1]
        encoder = StateEncoder().fit(grid, min_values, max_values)
        state = encoder.transform(-0.25, -0.25) # 0
        state = encoder.transform(0.25, 0.25) # 0
        state = encoder.transform(0.75, 0.75) # 3
        state = encoder.transform(1.25, 1.25) # 3
    """
    def __init__(self):
        """Initializes the StateEncoder object to a default state.

        Args:
            None

        Returns:
            None

        Raises:
            None
        """
        self._grid = None
        self._min_values = None
        self._max_values = None
        self._factors = None

    def fit(self, grid, min_values, max_values):
        """Generates the mapping from n-dim real space to discrete values.

        To omit a dimension entirely, set the number of grid elements  for that
        dimension to 1.

        Args:
            grid: An n-dimensional array on integers representing the number of
                grid intervals for each dimension.
            min_values: An n-dimensional array of floating point numbers
                representing the lower boundary for a given dimension.  All
                values less than the lower boundary will be "clamped" to the
                lower boundary.
            min_values: An n-dimensional array of floating point numbers
                representing the upper boundary for a given dimension.  All
                values greater than the upper boundary will be "clamped" to the
                upper boundary.

        Returns:
            None

        Raises:
            Exception when dimensions of grid, min_values, and max_values do
                not agree.
        """
        if len(grid) != len(min_values):
             raise Exception("grid and minimum values arrays have different dimensions")
        if len(grid) != len(max_values):
             raise Exception("grid and maximum values arrays have different dimensions")

        for dim, (min_val, max_val) in enumerate(zip(min_values, max_values)):
             if min_val > max_val:
                 raise Exception(f"Minimum value {min_val} is greater than maximum value {max_val} for dimension {dim}")
             if min_val == max_val:
                 raise Exception(f"Minimum value {min_val} is equal to maximum value {max_val} for dimension {dim}; to omit this dimension, use grid value of 1 instead")

        self._grid = grid[:]
        self._min_values = min_values[:]
        self._max_values = max_values[:]
        # Generate the factors used for the polynomial expansion, i.e. for a
        # [10, 10, 10] grid the factors would be [1, 10, 100]
        self._factors = np.array([1], dtype="int")
        for item in self._grid[:-1]:
            next_factor = item * self._factors[-1]
            self._factors = np.append(self._factors, next_factor)
        return self

    def transform(self, coords):
        """Transforms a point in n-dim real space to a integer value.

        The fit() function must be called to set up the mapping before this
        function is called.

        Args:
            coords: An n-dimensional array of floating point numbers
                representing the current state in an n-dimentional continuous
                space.

        Returns:
            An integer representing the state in discrete space.

        Raises:
            None
        """
        if len(self._grid) != len(coords):
             raise Exception("Dimension of real-space coordinates differs from the encoder's grid")

        factor = 1
        transformed = np.array([], dtype="int")
        for i, value in enumerate(coords):
            min_value = self._min_values[i]
            max_value = self._max_values[i]
            grid = self._grid[i]
            if value <= min_value:
                index = 0
            elif value >= max_value:
                index = grid - 1
            else:
                index = floor(
                    grid * (value - min_value) / (max_value - min_value)
                )
            transformed = np.append(transformed, index)
        return np.dot(self._factors, transformed)
