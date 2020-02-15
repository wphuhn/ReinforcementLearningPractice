import gym
from gym import wrappers

def print_run_summary(elapsed_time, run_index, n_steps, cumul_reward):
    print("Elapsed Time: {} , Run: {} , Number Steps: {} , Final Score: {}".format(elapsed_time, run_index, n_steps, cumul_reward))

def print_step_statistics(timestep, reward, cumul_reward, info):
    print("Step {} , Current reward {} , Cummul Reward {} , Lives Left {}".format(timestep + 1, reward, cumul_reward, info["ale.lives"]))

def make_envs(env_name, output_movie=False, output_folder=None):
    if output_movie:
        env_raw = gym.make(env_name)
        env = wrappers.Monitor(env_raw, output_folder, force=True)
    else:
        env_raw = None
        env = gym.make(env_name)
    env.reset()
    return env, env_raw

def close_envs(env, env_raw):
    env.close()
    if env_raw is not None:
        env_raw.close()
