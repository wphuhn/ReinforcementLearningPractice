import pickle as pkl
from time import time

import numpy as np
import torch

# TODO: Get rid of "device" throughout code, doesn't belong here


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
        self._time_current_qvals = 0
        self._time_target_qvals = 0
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
        self._time_current_qvals = 0
        self._time_target_qvals = 0
        self._time_train = 0
        self._time_target_update = 0

    def _output_timings(self):
        print(f"- init time {self._time_init} s")
        print(f"- act time {self._time_act} s")
        print(f"- replay time {self._time_replay} s")
        print(f"- batch create time {self._time_batch_create} s")
        print(f"- current qvals time {self._time_current_qvals} s")
        print(f"- target qvals time {self._time_target_qvals} s")
        print(f"- train time {self._time_train} s")
        print(f"- target update time {self._time_target_update} s")
        print(f"- epoch time {self._time_episode} s")

    def _optimize_model(self, net, target_net, criterion, optimizer, replay, batch_size, device):
        temp_time = time()
        mini_batch = replay.create_mini_batch(batch_size=batch_size)
        self._time_batch_create += time() - temp_time

        temp_time = time()
        current_qval = net.get_current_q_vals(mini_batch, device)
        self._time_current_qvals += time() - temp_time

        temp_time = time()
        target_qval = target_net.get_target_q_vals(mini_batch, device, self.gamma)
        self._time_target_qvals += time() - temp_time

        # Train model
        temp_time = time()
        loss = net.train_using_target_qvals(current_qval, target_qval, criterion, optimizer)
        self._time_train += time() - temp_time
        return loss

    def train(self, game, net, criterion, optimizer, device, replay, target_update_freq=10000, save_freq=500,
              batch_size=32):
        game.create()

        while self.timestep < self.n_steps:
            reward_episode = 0
            n_random = 0
            n_maximal = 0
            actions_taken = {}
            self._reset_episode_timers()
            temp_time = time()

            target_net = net.create_target_network(device)

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
                    n_random += 1
                else:
                    action = net.generate_best_action(state, device)
                    n_maximal += 1

                if action not in actions_taken:
                    actions_taken[action] = 0
                actions_taken[action] += 1

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
                loss = self._optimize_model(net, target_net, criterion, optimizer, replay, batch_size, device)
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
            print(f"- # random actions:  {n_random}")
            print(f"- # maximal actions: {n_maximal}")
            print(f"- actions taken:     {actions_taken}")
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

    # TODO:  Remove PyTorch-specific implementation
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
