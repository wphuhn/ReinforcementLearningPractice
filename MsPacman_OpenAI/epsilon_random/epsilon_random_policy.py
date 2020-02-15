from time import time, sleep

import gym
from gym import wrappers

from rl_functions.policies import next_action_with_epsilon_random_policy
from rl_functions.utilities import print_run_summary

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
        env_raw = gym.make(ENV_NAME)
        if OUTPUT_MOVIE:
            env = wrappers.Monitor(env_raw, OUTPUT_FOLDER, force=True)
        else:
            env = env_raw
        env.reset()

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
                print("Step {} , Current reward {} , Cummul Reward {} , Lives Left {}".format(timestep + 1, reward, cumul_reward, info["ale.lives"]))
            if done:
                elapsed_time = time() - start_time
                print_run_summary(elapsed_time, run_index + 1, timestep + 1, cumul_reward)
                break

        env.close()
        if OUTPUT_MOVIE:
            env_raw.close()
            break

if __name__ == "__main__":
    main()
