from time import time, sleep

import gym
from gym import wrappers

from rl_functions.policies import next_action_with_epsilon_random_policy
from rl_functions.utilities import print_run_summary, make_envs, close_envs, \
     print_step_statistics

# Roughly 30 minutes runtime on my laptop
EPSILON = 0.05
ENV_NAME = 'MsPacman-v0'
MAX_STEPS_PER_RUN = 10000000
NUM_RUNS = 4200 # Only relevant when OUTPUT_MOVIE = False
OUTPUT_MOVIE = False
FRAME_RATE = 1./30. # 30 FPS rendering
OUTPUT_FOLDER = "./videos/" + str(time()) + "/"

def main():
    start_time = time()
    for run_index in range(NUM_RUNS):
        env, env_raw = make_envs(ENV_NAME, output_movie=OUTPUT_MOVIE, output_folder=OUTPUT_FOLDER)

        cumul_reward = 0.0
        action = None
        for timestep in range(MAX_STEPS_PER_RUN):
            if OUTPUT_MOVIE:
                env.render()
                sleep(FRAME_RATE)

            action = next_action_with_epsilon_random_policy(env, EPSILON, action)
            _, reward, done, info = env.step(action)
            cumul_reward += reward

            if OUTPUT_MOVIE:
                print_step_statistics(timestep, reward, cumul_reward, info)
            if done:
                elapsed_time = time() - start_time
                print_run_summary(elapsed_time, run_index + 1, timestep + 1, cumul_reward)
                break

        close_envs(env, env_raw)
        if OUTPUT_MOVIE:
            break

if __name__ == "__main__":
    main()
