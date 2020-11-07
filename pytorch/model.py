from collections import deque
import random

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class AtariModel(nn.Module):

    def __init__(self, n_frames=4, n_outputs=4):
        super(AtariModel, self).__init__()
        self.n_frames = n_frames
        self.n_outputs = n_outputs
        self.conv_1 = nn.Conv2d(self.n_frames, 32, 8, stride=4)
        self.conv_2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv_3 = nn.Conv2d(64, 64, 3, stride=1)
        self.linear_1 = nn.Linear(64 * 7 * 7, 512)
        self.linear_2 = nn.Linear(512, self.n_outputs)

    def forward(self, x):
        y = F.relu(self.conv_1(x))
        y = F.relu(self.conv_2(y))
        y = F.relu(self.conv_3(y))
        y = y.view(y.size(0), -1)
        y = F.relu(self.linear_1(y))
        return self.linear_2(y)

    def base_model(self):
        return AtariModel(self.n_frames, self.n_outputs)
