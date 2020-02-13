# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 19:58:59 2019

@author: will
"""

import gym
from gym import wrappers
import time
import random

env_to_wrap = gym.make("MsPacman-v0")
env = wrappers.Monitor(env_to_wrap, './videos/' + str(time.time()), force = True)
env.reset()

PROB_TO_REMAIN = 0.95
MAX_NUMBER_STEPS = 5000

# Pick the initial action
action = env.action_space.sample()

for i in range(MAX_NUMBER_STEPS):
    env.render()
    
    # Every step (including the first), roll the dice to determine if we 
    # reroll the action
    # Note that this can occur on the first time step and, even if a reroll
    # happens, we may randomly select the same action as before
    if random.random() > PROB_TO_REMAIN:
        action = env.action_space.sample()

    _, _, done, info = env.step(action)
    
    if done:
        break
    
env.close()
env_to_wrap.close()
