"""Miscellaneous helper subroutines that haven't found a place elsewhere.
"""
from time import time

import gym
from gym import wrappers

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
