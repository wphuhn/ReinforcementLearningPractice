import gym
from gym import wrappers
from time import time, sleep

env_to_wrap = gym.make('MsPacman-v0')
env = wrappers.Monitor(env_to_wrap, "./videos/" + str(time()) + "/", force = True)
env.reset()

cummul_reward = 0.0
for t in range(5000):
    # 30 FPS rendering
    env.render()
    sleep(1./30.)

    # Do the thing
    action = env.action_space.sample()
    _, reward, done, info = env.step(action)

    cummul_reward += reward
    print("Step {} , Current reward {} , Cummul Reward {} , Lives Left {}".format(t+1, reward, cummul_reward, info["ale.lives"]))
    if done:
        print()
        print("Number Steps: {} , Final Score: {}".format(t+1, cummul_reward))
        break

env.close()
env_to_wrap.close()
