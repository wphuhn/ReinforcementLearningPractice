from collections import deque
import random

import cv2
import gym
import matplotlib.pyplot as plt
import numpy as np
import torch


# TODO:  too much of the implementation leaks out into other classes (especially number of frames)
class State(object):

    def __init__(self, n_frames=4, frame_size=(84, 84)):
        self.n_frames = n_frames
        self.frame_size = frame_size
        self.frames = None

    def create(self, buffer):
        # The buffer has n_frames+1 frames, as it performs smoothing

        # Each pseudo-frame in the state is the maximum of the channel values
        # between a frame and the previous frame in the buffer (this is done to
        # handle flickering)
        rolling_maxs = []
        for i in range(self.n_frames):
            rolling_maxs.append(np.max([buffer[i + 1], buffer[i]], axis=0))
        '''
        rolling_maxs.append(np.max([oldest, buffer[0]], axis=0))    
        for i, frame in enumerate(buffer):
            if i > 0:
                rolling_maxs.append(np.max([buffer[i-1], frame], axis=0))
        '''

        # Convert RGB to luminance
        # Note: I am assuming that the RBG values output by OpenAI Gym are
        # already linear
        lums = []
        for frame in rolling_maxs:
            lums.append(0.2126 * frame[:, :, 0] + 0.7152 * frame[:, :, 1] + 0.0722 * frame[:, :, 2])

        # Normalize and resize frames to target size
        self.frames = []
        for frame in lums:
            frame /= 255.
            self.frames.append(cv2.resize(frame, dsize=self.frame_size))

        self.frames = np.array(self.frames)
        return self

    def plot(self):
        fig, axs = plt.subplots(2, 2, figsize=(20, 20))
        axs[0, 0].imshow(self.frames[3], cmap=plt.cm.binary)
        axs[0, 0].set_title("t")
        axs[0, 1].imshow(self.frames[2], cmap=plt.cm.binary)
        axs[0, 1].set_title("t-1")
        axs[1, 0].imshow(self.frames[1], cmap=plt.cm.binary)
        axs[1, 0].set_title("t-2")
        axs[1, 1].imshow(self.frames[0], cmap=plt.cm.binary)
        axs[1, 1].set_title("t-4")


class Game(object):

    def __init__(self, rom="Breakout-v0"):
        self.env = None
        self.frame = None
        self.n_actions = None
        self.rom = rom

    def create(self):
        if self.env is not None:
            self.close()
        self.env = gym.envs.make(self.rom)
        self.frame = self.env.reset()
        self.n_actions = self.env.action_space.n
        return self

    def get_action_meanings(self):
        return self.env.unwrapped.get_action_meanings()

    def sample(self):
        return self.env.action_space.sample()

    def step(self, action):
        frame, reward, done, info = self.env.step(action)
        self.frame = frame
        return frame, reward, done, info

    def step_state(self, action, n_frames=4, frame_size=(84, 84)):
        buffer = deque()
        buffer.append(self.frame)

        reward = 0
        for i in range(n_frames):
            frame, r, done, info = self.step(action)
            reward += r
            buffer.append(frame)
        return State(n_frames, frame_size).create(buffer), reward, done, info

    def reset(self):
        frame = self.env.reset()
        self.frame = frame
        return frame

    def close(self):
        self.env.render()
        self.env.close()
        self.env = None
        self.frame = None
        self.n_actions = None


class ReplayBuffer(object):
    # DeepMind 2015 used last 1 million frames, which corresponds to 250,000 states for 4-frame states
    def __init__(self, max_len=250000):
        self.replay = None
        self.max_len = max_len

    def add(self, state, action, reward, next_state, done):
        # Takes in a collection of Python/numpy primitives and converts them to Tensors
        # before appending to the replay buffer
        state_tensor = torch.tensor(state.frames, dtype=torch.half)
        action_tensor = torch.tensor([action], dtype=torch.uint8)
        reward_tensor = torch.tensor([reward], dtype=torch.half)
        next_state_tensor = torch.tensor(next_state.frames, dtype=torch.half)
        done_tensor = torch.tensor([done], dtype=torch.uint8)

        self.replay.append((state_tensor, action_tensor, reward_tensor, next_state_tensor, done_tensor))

    def populate(self, game, n_states=12500):
        self.replay = deque(maxlen=self.max_len)
        game.create()

        # Create initial state
        action = game.sample()
        state, _, _, _ = game.step_state(action)

        for step in range(n_states):
            action = game.sample()
            next_state, reward, done, _ = game.step_state(action)

            self.add(state, action, reward, next_state, done)

            if step % 1000 == 1:
                print(f"On step {step} of replay buffer")

            if done:
                _ = game.reset()

            state = next_state

        return self

    def create_mini_batch(self, batch_size=32):
        # Generate mini-batch
        mini_batch = random.sample(self.replay, batch_size)

        tensors = dict()
        tensors["action"] = torch.stack([a for (s, a, r, s_n, d) in mini_batch]).squeeze()
        tensors["reward"] = torch.stack([r for (s, a, r, s_n, d) in mini_batch]).squeeze()
        tensors["done"] = torch.stack([d for (s, a, r, s_n, d) in mini_batch]).squeeze()
        tensors["state"] = torch.stack([s for (s, a, r, s_n, d) in mini_batch])
        tensors["next_state"] = torch.stack([s_n for (s, a, r, s_n, d) in mini_batch])

        return tensors

    def __getitem__(self, key):
        return self.replay[key]

    def __setitem__(self, key, value):
        self.replay[key] = value
        return self.replay[key]
