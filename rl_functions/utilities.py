from time import time

import gym
from gym import wrappers

def run_summary(elapsed_time, run_index, n_steps, cumul_reward):
    return f"Elapsed Time: {elapsed_time} , Run: {run_index} , Number Steps: {n_steps} , Final Score: {cumul_reward}"

def step_statistics(timestep, reward, cumul_reward, lives_left):
    return f"Step {timestep} , Current reward {reward} , Cumul Reward {cumul_reward} , Lives Left {lives_left}"

def make_envs(env_name, output_movie=False, output_folder=None, env_seed=None,
              env_action_seed=None, max_steps_per_episode=None):
    # Create the environments
    if output_movie:
        env_raw = gym.make(env_name)
        if output_folder is None:
            output_folder = "./videos/" + str(time()) + "/"
        env = wrappers.Monitor(env_raw, output_folder, force=True)
    else:
        env_raw = None
        env = gym.make(env_name)
    # Set the random seeds
    if env_seed is not None:
        env.seed(env_seed)
    if env_action_seed is not None:
        # The action space random sampling does not use the same seed as the
        # environment
        env.action_space.np_random.seed(env_seed)
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

def close_envs(env, env_raw):
    env.close()
    if env_raw is not None:
        env_raw.close()
