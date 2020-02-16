from time import time

import gym
from gym import wrappers

def run_summary(elapsed_time, run_index, n_steps, cumul_reward):
    return "Elapsed Time: {} , Run: {} , Number Steps: {} , Final Score: {}".format(elapsed_time, run_index, n_steps, cumul_reward)

def step_statistics(timestep, reward, cumul_reward, lives_left):
    return "Step {} , Current reward {} , Cummul Reward {} , Lives Left {}".format(timestep, reward, cumul_reward, lives_left)

def make_envs(env_name, output_movie=False, output_folder=None, env_seed=None, env_action_seed=None):
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
    # Initialize the environment (does not change seed)
    env.reset()
    return env, env_raw

def close_envs(env, env_raw):
    env.close()
    if env_raw is not None:
        env_raw.close()
