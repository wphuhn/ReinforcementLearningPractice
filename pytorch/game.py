from collections import deque

import cv2
import gym
import matplotlib.pyplot as plt
import numpy as np


# TODO:  too much of the implementation leaks out into other classes (especially number of frames)
class State(object):

    def __init__(self, n_frames=4, frame_size=(84, 84)):
        self.n_frames = n_frames
        self.frame_size = frame_size
        self.frames = None

    def create(self, buffer):
        # The buffer has n_frames+1 frames, as it performs smoothing
        rolling_maxs = self.create_rolling_max(buffer, self.n_frames)
        lums = self.convert_rbg_to_lum(rolling_maxs)
        frames = self.normalize_frames(lums, self.frame_size)
        self.frames = np.array(frames)
        return self

    @staticmethod
    def normalize_frames(lums, frame_size):
        # Normalize and resize frames to target size
        frames = []
        for frame in lums:
            frame /= 255.
            frames.append(cv2.resize(frame, dsize=frame_size))
        return frames

    @staticmethod
    def convert_rbg_to_lum(rolling_maxs):
        # Convert RGB to luminance
        # Note: I am assuming that the RBG values output by OpenAI Gym are
        # already linear
        lums = []
        for frame in rolling_maxs:
            lums.append(0.2126 * frame[:, :, 0] + 0.7152 * frame[:, :, 1] + 0.0722 * frame[:, :, 2])
        return lums

    @staticmethod
    def create_rolling_max(buffer, n_frames):
        # Each pseudo-frame in the state is the maximum of the channel values
        # between a frame and the previous frame in the buffer (this is done to
        # handle flickering)
        rolling_maxs = []
        for i in range(n_frames):
            rolling_maxs.append(np.max([buffer[i + 1], buffer[i]], axis=0))
        '''
        rolling_maxs.append(np.max([oldest, buffer[0]], axis=0))    
        for i, frame in enumerate(buffer):
            if i > 0:
                rolling_maxs.append(np.max([buffer[i-1], frame], axis=0))
        '''
        return rolling_maxs

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
