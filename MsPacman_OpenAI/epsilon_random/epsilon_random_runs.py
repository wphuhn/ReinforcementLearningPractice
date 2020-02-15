import time
import random

import gym

from rl_functions.policies import next_action_with_epsilon_random_policy
from rl_functions.utilities import print_run_summary

# Roughly 30 minutes runtime on my laptop
NUM_RUNS = 4200
MAX_STEPS_PER_RUN = 10000000 # It'll never get close to this (famous last words)
ENV_NAME = 'MsPacman-v0'
EPSILON = 0.001

def main():
    start_time = time.time()
    for run_index in range(NUM_RUNS):
        env = gym.make(ENV_NAME)
        env.reset()

        cumul_reward = 0.0
        action = None
        for timestep in range(MAX_STEPS_PER_RUN):
            action = next_action_with_epsilon_random_policy(env, EPSILON, action)
            _, reward, done, _ = env.step(action)

            cumul_reward += reward
            if done:
                elapsed_time = time.time() - start_time
                print_run_summary(elapsed_time, run_index + 1, timestep + 1, cumul_reward)
                break

        env.close()

if __name__ == "__main__":
    main()
