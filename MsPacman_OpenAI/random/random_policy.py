from time import time, sleep

import gym
from gym import wrappers

from rl_functions.policies import next_action_with_random_policy
from rl_functions.utilities import run_summary, make_envs, close_envs, \
     step_statistics

# Roughly 30 minutes runtime on my laptop

ENV_NAME = 'MsPacman-v0'
MAX_STEPS_PER_RUN = 10000000
NUM_RUNS = 4200 # Only relevant when OUTPUT_MOVIE = False
OUTPUT_MOVIE = False
FRAME_RATE = 1./30. # 30 FPS rendering
N_ACTIONS = 8 # This should not be hard wired!

def main():
    start_time = time()
    for run_index in range(NUM_RUNS):
        env, env_raw = make_envs(ENV_NAME, output_movie=OUTPUT_MOVIE)

        cumul_reward = 0.0
        action = None
        for timestep in range(MAX_STEPS_PER_RUN):
            if OUTPUT_MOVIE:
                env.render()
                sleep(FRAME_RATE)

            action = next_action_with_random_policy(N_ACTIONS)
            _, reward, done, info = env.step(action)
            cumul_reward += reward

            if OUTPUT_MOVIE:
                print(step_statistics(timestep + 1, reward, cumul_reward, info['ale.lives']))
            if done:
                elapsed_time = time() - start_time
                print(run_summary(elapsed_time, run_index + 1, timestep + 1, cumul_reward))
                break

        close_envs(env, env_raw)
        if OUTPUT_MOVIE:
            break

if __name__ == "__main__":
    main()
