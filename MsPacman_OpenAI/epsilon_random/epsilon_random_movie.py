from time import time
import random

import gym
from gym import wrappers

from rl_functions.policies import next_action_with_epsilon_random_policy

MAX_STEPS_PER_RUN = 5000
ENV_NAME = 'MsPacman-v0'
#FRAME_RATE = _
OUTPUT_FOLDER = "./videos/" + str(time()) + "/"
EPSILON = 0.05

def main():
    env_to_wrap = gym.make(ENV_NAME)
    env = wrappers.Monitor(env_to_wrap, OUTPUT_FOLDER, force=True)
    env.reset()

    action = None
    for _ in range(MAX_STEPS_PER_RUN):
        env.render()

        action = next_action_with_epsilon_random_policy(env, EPSILON, action)
        _, _, done, _ = env.step(action)

        if done:
            break

    env.close()
    env_to_wrap.close()

if __name__ == "__main__":
    main()
