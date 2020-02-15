from time import time, sleep

import gym
from gym import wrappers

from rl_functions.policies import next_action_with_random_policy

MAX_STEPS_PER_RUN = 5000
ENV_NAME = 'MsPacman-v0'
FRAME_RATE = 1./30. # 30 FPS rendering
OUTPUT_FOLDER = "./videos/" + str(time()) + "/"

def main():
    env_to_wrap = gym.make(ENV_NAME)
    env = wrappers.Monitor(env_to_wrap, OUTPUT_FOLDER, force=True)
    env.reset()

    cumul_reward = 0.0
    for timestep in range(MAX_STEPS_PER_RUN):
        # 30 FPS rendering
        env.render()
        sleep(FRAME_RATE)

        # Do the thing
        action = next_action_with_random_policy(env)
        _, reward, done, info = env.step(action)

        cumul_reward += reward
        print("Step {} , Current reward {} , Cummul Reward {} , Lives Left {}".format(timestep + 1, reward, cumul_reward, info["ale.lives"]))
        if done:
            print()
            print("Number Steps: {} , Final Score: {}".format(timestep + 1, cumul_reward))
            break

    env.close()
    env_to_wrap.close()

if __name__ == "__main__":
    main()
