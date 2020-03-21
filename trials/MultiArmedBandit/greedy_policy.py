from time import time

import numpy as np
from numpy.random import normal

from rl_functions.policies import GreedyPolicy, RandomPolicy
from rl_functions.updates import IterativeControl
from rl_functions.utilities import run_summary

MAX_STEPS_PER_RUN = 100
NUM_RUNS = 4200
ALPHA = 0.1
EPSILON = 0.01

def multi_arm_bandit(action):
    std_dev = 0.1
    rewards = [0, 2, 5, 3, 4, -2, -5, -3, -4]
    return normal(rewards[action], std_dev)

def main():
    start_time = time()
    state = 0 # Multi-armed bandit has only one state
    # Initialize q-function optimistically for all actions
    q_function = {state:
        {key: value for key, value in zip(range(9), [100.]*9)}
    }
    greedy_policy = GreedyPolicy(epsilon=EPSILON, random_policy=RandomPolicy(9))
    control = IterativeControl(alpha=ALPHA, q=q_function)
    for run_index in range(NUM_RUNS):
        cumul_reward = 0.0
        for timestep in range(MAX_STEPS_PER_RUN):
            action = greedy_policy.next_action(q_function, state)
            reward = multi_arm_bandit(action)
            trajectory = [(state, action)]
            rewards = [reward]
            control.update(trajectory, rewards)
            q_function = control.get_q()
            cumul_reward += reward
        print(run_summary(0, run_index + 1, timestep + 1, cumul_reward))
        if run_index % 100 == 0:
            print(f"Run {run_index + 1} : {q_function}")
    elapsed_time = time() - start_time
    print(f"Elapsed time : {elapsed_time} s")

if __name__ == "__main__":
    main()
