import pickle as pkl
from time import time

import numpy as np
import torch


class QLearning(object):

    def __init__(self, output_folder=None, max_steps_per_eps=50000, n_steps=50000000, gamma=0.99):
        self.output_folder = output_folder
        self.max_steps_per_eps = max_steps_per_eps
        self.n_steps = n_steps
        self.gamma = gamma

        self.timestep = 0
        self.epoch = 0
        self.losses = []
        self.rewards_episode = []

        # TODO: Too many variables
        self._time_init = 0
        self._time_act = 0
        self._time_replay = 0
        self._time_batch_create = 0
        self._time_batch_transfer = 0
        self._time_qs = 0
        self._time_train = 0
        self._time_target_update = 0
        self._time_episode = 0

    @staticmethod
    def epsilon_schedule(step, n_steps=250000, eps_max=1.0, eps_min=0.1):
        """
        Linear anneal schedule

        Taken from Mnih et al. 2013
        """
        if step < 1:
            return eps_max
        if step > n_steps:
            return eps_min
        return (eps_min - eps_max) / (n_steps - 1) * (step - 1) + eps_max

    def _reset_episode_timers(self):
        self._time_episode = time()

        self._time_act = 0
        self._time_replay = 0
        self._time_batch_create = 0
        self._time_batch_transfer = 0
        self._time_qs = 0
        self._time_train = 0
        self._time_target_update = 0

    def _output_timings(self):
        print(f"- init time {self._time_init} s")
        print(f"- act time {self._time_act} s")
        print(f"- replay time {self._time_replay} s")
        print(f"- batch create time {self._time_batch_create} s")
        print(f"- batch transfer time {self._time_batch_transfer} s")
        print(f"- qs time {self._time_qs} s")
        print(f"- train time {self._time_train} s")
        print(f"- target update time {self._time_target_update} s")
        print(f"- epoch time {self._time_episode} s")

    def train(self, game, net, criterion, optimizer, device, replay, target_update_freq=10000, save_freq=500,
              batch_size=32):
        game.create()

        while self.timestep < self.n_steps:
            reward_episode = 0
            self._reset_episode_timers()
            temp_time = time()

            target_net = self._create_target_network(net, device)

            _ = game.reset()
            # TODO: Should do a proper first step, not a random initialization
            action = game.sample()
            state, _, _, _ = game.step_state(action)

            self._time_init = time() - temp_time
            for step in range(self.max_steps_per_eps):
                temp_time = time()

                # Epsilon scheduling
                epsilon = self.epsilon_schedule(self.timestep)
                if np.random.random() < epsilon:
                    action = game.sample()
                else:
                    action = self._generate_best_action(net, state, device)

                # Run environment and create next state
                # To accelerate performance, we repeat the same action repeatedly
                # for a fixed number of steps
                next_state, reward, done, _ = game.step_state(action)
                reward_episode += reward
                self._time_act += time() - temp_time

                temp_time = time()
                replay.add(state, action, reward, next_state, done)
                self._time_replay += time() - temp_time

                # Update model
                loss = self._train_model(net, target_net, criterion, optimizer, replay, batch_size, device)
                self.losses.append(loss)

                # Update target model
                temp_time = time()
                if self.timestep % target_update_freq == 0:
                    target_net.set_to(net)
                self._time_target_update += time() - temp_time

                self.timestep += 1

                # Move to next iteration
                if done:
                    break
                state = next_state
            self.rewards_episode.append(reward_episode)
            self._time_episode = time() - self._time_episode

            print(f"Finished epoch {self.epoch}, total rewards {reward_episode}, total num steps {step}, buffer length {len(replay.replay)}, final epsilon {epsilon}")
            self._output_timings()

            if self.output_folder is not None:
                if self.epoch % save_freq == 0:
                    time_save = time()
                    self.save(game, net, criterion, optimizer, replay, suffix=f".latest")
                    time_save = time() - time_save
                    print(f"Finished saving epoch {self.epoch}, total time to save {time_save} s")

            self.epoch += 1

        self.save(game, net, criterion, optimizer, replay, suffix=f".latest")

        return self.losses, self.rewards_episode, replay

    @staticmethod
    def _create_target_network(net, device):
        # Set up target network
        target_net = net.base_model()
        target_net.half()
        target_net.to(device)
        target_net.load_state_dict(net.state_dict())
        return target_net

    @staticmethod
    def _generate_best_action(net, state, device):
        # Generate action
        q_values = net(torch.tensor(state.frames, dtype=torch.half, device=device).unsqueeze(0))
        # Needs to reside on CPU to be fed to OpenAI Gym, and argmax doesn't accept half precision
        q_values = q_values.clone().detach().float().cpu()
        # The typecast is needed to handle an edge case:
        # numpy() returns a 0D ndarray, which will cause the mini-batch
        # construction to throw an "object does not have length" error
        # if it's the first element in the mini-batch
        return int(torch.argmax(q_values).data.numpy())

    def _train_model(self, net, target_net, criterion, optimizer, replay, batch_size, device):
        temp_time = time()
        mini_batch = replay.create_mini_batch(batch_size=batch_size)
        self._time_batch_create += time() - temp_time

        temp_time = time()
        state_batch = mini_batch["state"].to(device)
        action_batch = mini_batch["action"].to(device)
        reward_batch = mini_batch["reward"].to(device)
        next_state_batch = mini_batch["next_state"].to(device)
        done_batch = mini_batch["done"].to(device)
        self._time_batch_transfer += time() - temp_time

        temp_time = time()
        # Get model predicted q values
        cur_q = net(state_batch)
        # Get target-model predicted q values
        with torch.no_grad():
            # torch.max isn't implemented for half precision, so use single
            next_q = target_net(next_state_batch).float()

        # Get the expected and target q-values
        current_qval = cur_q.gather(dim=1, index=action_batch.long().unsqueeze(dim=1)).squeeze()
        target_qval = reward_batch + self.gamma * ((1 - done_batch) * torch.max(next_q, dim=1)[0])
        # Convert back to half precision for optimizer
        target_qval = target_qval.half()
        self._time_qs += time() - temp_time

        # Train model
        temp_time = time()
        loss = criterion(current_qval, target_qval.detach())
        optimizer.zero_grad()
        loss.backward()
        # Perform gradient clipping:
        for param in net.parameters():
            param.grad.clamp(-1, 1)
        optimizer.step()
        self._time_train += time() - temp_time
        return loss.item()

    def save(self, game, net, criterion, optimizer, replay, save_replay=False, suffix=""):
        with open(f"{self.output_folder}/training{suffix}.pkl", "wb") as training_file:
            pkl.dump(self, training_file)
        with open(f"{self.output_folder}/game{suffix}.pkl", "wb") as game_file:
            pkl.dump(game, game_file)
        with open(f"{self.output_folder}/net{suffix}.pth", "wb") as net_file:
            torch.save(net.state_dict(), net_file)
        with open(f"{self.output_folder}/criterion{suffix}.pth", "wb") as criterion_file:
            torch.save(criterion.state_dict(), criterion_file)
        with open(f"{self.output_folder}/optimizer{suffix}.pth", "wb") as optimizer_file:
            torch.save(optimizer.state_dict(), optimizer_file)
        # Saving the replay buffer via Pickle can lead to OOM, as Pickle creates a copy
        # of the save in a VM
        # What *should* be done is to not use Pickle at all, but as a stop-gap measure,
        # I'm disabling replay buffer saving by default
        if save_replay:
            with open(f"{self.output_folder}/replay{suffix}.pkl", "wb") as replay_file:
                pkl.dump(replay, replay_file)
