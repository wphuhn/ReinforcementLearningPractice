import gym
import time
import random

# Roughly 30 minutes runtime on my laptop
MAX_STEPS_PER_RUN = 10000000 # It'll never get close to this (famous last words)
NUM_RUNS = 4200
PROB_TO_REMAIN = 0.999

start_time = time.time()
print("Run ElapsedTime NumberSteps CumulScore")
for r in range(NUM_RUNS):
    env = gym.make('MsPacman-v0')
    env.reset()

    cumul_reward = 0.0
    # Pick the initial action
    action = env.action_space.sample()
    for t in range(MAX_STEPS_PER_RUN):
        if random.random() > PROB_TO_REMAIN:
            action = env.action_space.sample()
        _, reward, done, info = env.step(action)

        cumul_reward += reward
        if done:
            elapsed_time = time.time() - start_time
            print("{} {} {} {}".format(r+1, elapsed_time, t+1, cumul_reward))
            break

    env.close()
