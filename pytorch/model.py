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

    def set_to(self, net):
        self.load_state_dict(net.state_dict())

    def create_target_network(self, device):
        # Set up target network
        target_net = AtariModel(self.n_frames, self.n_outputs)
        target_net.half()
        target_net.to(device)
        target_net.load_state_dict(self.state_dict())
        return target_net

    def generate_best_action(self, state, device):
        # Generate action
        q_values = self(torch.tensor(state.frames, dtype=torch.half, device=device).unsqueeze(0))
        # Needs to reside on CPU to be fed to OpenAI Gym, and argmax doesn't accept half precision
        q_values = q_values.clone().detach().float().cpu()
        # The typecast is needed to handle an edge case:
        # numpy() returns a 0D ndarray, which will cause the mini-batch
        # construction to throw an "object does not have length" error
        # if it's the first element in the mini-batch
        return int(torch.argmax(q_values).data.numpy())

    def get_current_q_vals(self, mini_batch, device):
        state_batch = mini_batch["state"].to(device)
        action_batch = mini_batch["action"].to(device)
        # Get model predicted q values
        cur_q = self(state_batch)
        current_qval = cur_q.gather(dim=1, index=action_batch.long().unsqueeze(dim=1)).squeeze()
        return current_qval

    def get_target_q_vals(self, mini_batch, device, gamma):
        next_state_batch = mini_batch["next_state"].to(device)
        reward_batch = mini_batch["reward"].to(device)
        done_batch = mini_batch["done"].to(device)
        # Get target-model predicted q values
        with torch.no_grad():
            # torch.max isn't implemented for half precision, so use single
            next_q = self(next_state_batch).float()
        target_qval = reward_batch + gamma * ((1 - done_batch) * torch.max(next_q, dim=1)[0])
        # Convert back to half precision for optimizer
        target_qval = target_qval.half()
        return target_qval

    def train_using_target_qvals(self, current_qval, target_qval, criterion, optimizer):
        loss = criterion(current_qval, target_qval.detach())
        optimizer.zero_grad()
        loss.backward()
        # Perform gradient clipping:
        for param in self.parameters():
            param.grad.clamp(-1, 1)
        optimizer.step()
        return loss.item()
