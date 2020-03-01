from time import time, sleep

import gym
from gym import wrappers

from rl_functions.policies import ConstantPolicy, RandomPolicy
from rl_functions.utilities import run_summary, make_envs, close_envs, \
     step_statistics

# Roughly 30 minutes runtime on my laptop
EPSILON = 0.05
ENV_NAME = 'MsPacman-v0'
MAX_STEPS_PER_RUN = 10000000
NUM_RUNS = 4200 # Only relevant when OUTPUT_MOVIE = False
OUTPUT_MOVIE = False
FRAME_RATE = 1./30. # 30 FPS rendering

def main():
    start_time = time()

    n_actions = gym.make(ENV_NAME).action_space.n
    random_policy = RandomPolicy(n_actions)
    action = random_policy.next_action()
    constant_policy = ConstantPolicy(action, epsilon=EPSILON,
        random_policy=random_policy)
    for run_index in range(NUM_RUNS):
        env, env_raw, _ = make_envs(ENV_NAME, output_movie=OUTPUT_MOVIE)

        cumul_reward = 0.0
        for timestep in range(MAX_STEPS_PER_RUN):
            if OUTPUT_MOVIE:
                env.render()
                sleep(FRAME_RATE)

            action = constant_policy.next_action()
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
