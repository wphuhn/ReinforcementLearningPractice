from time import time

import numpy as np
from numpy.random import normal

from rl_functions.policies import RandomPolicy
from rl_functions.controls import IterativeControl
from rl_functions.utilities import run_summary

MAX_STEPS_PER_EPISODE = 100
N_EPISODES = 4200
ALPHA = 0.1
EPSILON = 0.01
DUMMY_STATE = 0

class MultiArmBandit(object):
    @staticmethod
    def step(action):
        std_dev = 0.1
        rewards = [0, 2, 5, 3, 4, -2, -5, -3, -4]
        reward = normal(rewards[action], std_dev)
        return DUMMY_STATE, reward, False, ""

def main():
    start_time = time()
    initial_state = DUMMY_STATE # Multi-armed bandit has only one state
    # Initialize q-function optimistically for all actions
    q_function = {initial_state:
        {key: value for key, value in zip(range(9), [100.]*9)}
    }
    control = IterativeControl(
        alpha=ALPHA,
        epsilon=EPSILON,
        random_policy=RandomPolicy(9),
        q=q_function,
    )
    env = MultiArmBandit()
    for episode_index in range(N_EPISODES):
        trajectory, rewards, infos = control.run_episode(
            env,
            initial_state,
            max_n_steps=MAX_STEPS_PER_EPISODE,
        )
        cumul_reward = sum(rewards)
        print(run_summary(0, episode_index + 1, MAX_STEPS_PER_EPISODE, cumul_reward))
        if episode_index % 100 == 0:
            q_function = control.get_q()
            print(f"Run {episode_index + 1} : {q_function}")
    elapsed_time = time() - start_time
    print(f"Elapsed time : {elapsed_time} s")

if __name__ == "__main__":
    main()
