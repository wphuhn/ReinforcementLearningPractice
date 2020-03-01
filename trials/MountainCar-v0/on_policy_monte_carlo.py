import pickle
from time import time, sleep

import gym

from rl_functions.policies import RandomPolicy, GreedyPolicy
from rl_functions.updates import update_q_on_policy_monte_carlo
from rl_functions.utilities import run_summary, make_envs, close_envs, \
     step_statistics

# Various run-time parameters to be tweaked
EPSILON = 0.05 # Degree of randomness in epsilon policy when training
ENV_NAME = 'MountainCar-v0' # OpenAI environment
GAMMA = 0.9 # Discount factor for Monte Carlo value estimation
GRID = [100, 100] # Discretization grid for environment
MAX_STEPS_PER_EPISODE = 10000000 # Number of steps per episode (obviously)
NUM_EPSIODES = 4200 # Number of episodes to run
OUTPUT_MOVIE = False # Whether to output a movie (will not train when doing so)
FRAME_RATE = 1./30. # FPS when rending a movie

def map_2D_to_discrete_space(coord, grid, min_values, max_values):
    # Maps the real-valued 2D observation state returned by the environment to
    # a discretized integer-valued state
    from math import floor
    x = grid[0] * (coord[0] - min_values[0]) / (max_values[0] - min_values[0])
    y = grid[1] * (coord[1] - min_values[1]) / (max_values[1] - min_values[1])
    return floor(y) * grid[0] + floor(x)

def main():
    start_time = time()

    # Extract the features of the environment
    n_states = GRID[0] * GRID[1]
    n_actions = gym.make(ENV_NAME).action_space.n
    min_values = gym.make(ENV_NAME).observation_space.low
    max_values = gym.make(ENV_NAME).observation_space.high

    q = {}
    counts = {}
    if OUTPUT_MOVIE:
        # If we're making a movie, we create a (truly) greedy policy and load
        # a pre-existing q function
        policy = GreedyPolicy()
        with open("best_q.pkl", "rb") as q_file:
            q = pickle.load(q_file)
    else:
        # If we're training a q function, we load an epsilon-soft greedy policy
        # and initialize the q function and counts to all zeros.  Since the
        # greedy policy breaks ties randomly, this is essentially a random
        # policy stating out
        policy = GreedyPolicy(
            epsilon=EPSILON,
            random_policy=RandomPolicy(n_actions),
        )
        for state in range(n_states):
            q[state] = {key: 0. for key in range(n_actions)}
            counts[state] = {key: 0 for key in range(n_actions)}

    # Outer loop for episode
    for episode in range(NUM_EPSIODES):
        # Create the initial environment and register the initial state
        env, env_raw, obs = make_envs(
            ENV_NAME,
            output_movie=OUTPUT_MOVIE,
            max_steps_per_episode=MAX_STEPS_PER_EPISODE,
        )
        state = map_2D_to_discrete_space(obs, GRID, min_values, max_values)

        # Generate the trajectory for this episode
        trajectory = []
        rewards = []
        cumul_reward = 0.
        for timestep in range(MAX_STEPS_PER_EPISODE):
            # Render the current frame
            if OUTPUT_MOVIE:
                env.render()
                sleep(FRAME_RATE)

            # Predict the current action and generate the reward
            action = policy.next_action(q, state)
            trajectory.append((state, action))
            observation, reward, done, _ = env.step(action)
            rewards.append(reward)
            cumul_reward += reward

            # Output time step results to screen.  Only do this when outputting
            # movies to avoid slowing down the training.
            if OUTPUT_MOVIE:
                print(step_statistics(timestep + 1, reward, cumul_reward, state))
            # If we're finished with this episode, output episode statistics
            # and break
            if done:
                elapsed_time = time() - start_time
                print(run_summary(elapsed_time, episode + 1, timestep + 1, cumul_reward))
                break
            # Get the new state, if we haven't
            state = map_2D_to_discrete_space(
                observation,
                GRID,
                min_values,
                max_values,
            )

        # Update the q function based on the trajector for this episode (when
        # training)
        if not OUTPUT_MOVIE:
            q, counts = update_q_on_policy_monte_carlo(
                trajectory,
                rewards,
                q,
                counts,
                GAMMA,
            )

        close_envs(env, env_raw)
        if OUTPUT_MOVIE:
            break
        # If the current q function gives the highest cumulative score, save it
        # to disk
        if episode == 0:
            with open("best_q.pkl", "wb") as q_file:
                pickle.dump(q, q_file)
            max_cumul_reward = cumul_reward
        elif max_cumul_reward < cumul_reward:
            with open("best_q.pkl", "wb") as q_file:
                pickle.dump(q, q_file)

if __name__ == "__main__":
    main()
