import gym
import time

# Roughly 30 minutes runtime on my laptop
NUM_RUNS = 4200
MAX_STEPS_PER_RUN = 10000000 # It'll never get close to this (famous last words)

start_time = time.time()

for r in range(NUM_RUNS):
    env = gym.make('MsPacman-v0')
    env.reset()

    cummul_reward = 0.0
    for t in range(MAX_STEPS_PER_RUN):
        # Do the thing
        action = env.action_space.sample()
        _, reward, done, info = env.step(action)

        cummul_reward += reward
        if done:
            elapsed_time = time.time() - start_time
            print("Elapsed Time: {} , Run: {} , Number Steps: {} , Final Score: {}".format(elapsed_time, r+1, t+1, cummul_reward))
            break

    env.close()
