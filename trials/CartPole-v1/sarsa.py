import pickle
from time import time, sleep

import gym

from rl_functions.policies import RandomPolicy, GreedyPolicy
from rl_functions.controls import SarsaControl
from rl_functions.utilities import (
    run_summary,
    make_envs,
    close_envs,
    step_statistics,
    StateEncoder,
)

"""
Script for solving the CartPole-v1 environment using SARSA:

Observations:

1) By far the most important part of this environment is recognizing that the
   range of values allowable for the observables is *much* larger than the range
   of values that realistically matters.  While some of them are obviously
   overkill (+- infinity for speeds, for example), the range of values for
   angles that are achievable without failing is tiny.
2) The most common lose condition is the bar losing control (angular momentum
   large).  The second most common lose condition is the bar slowly tipping over
   (angle large).  The third most common lose condition is the bar being stable
   but the bar veering off.
3) A coarse grid is better than a fine grid, since with a fine grid, many states
   will only be visited once and the statistics will be bad.  Also, memory
   becomes an issue.
4) Setting an optimistic initial value actually made things worse, because it
   incentized exploration too much.
"""

# Various run-time parameters to be tweaked
EPSILON = 0.1 # Degree of randomness in epsilon policy when training
ENV_NAME = 'CartPole-v1' # OpenAI environment
ALPHA = 0.1 # Update factor
GAMMA = 1. - 1./200.0 # Discount factor for value estimation
GRID = [1, 10, 10, 10] # Discretization grid for environment
INITIAL_VALUE = 10 # Initial value for q-function
MAX_STEPS_PER_EPISODE = 200 # Number of steps per episode (obviously)
NUM_EPSIODES = 420000 # Number of episodes to run
OUTPUT_MOVIE = False # Whether to output a movie (will not train when doing so)
START_FROM_MODEL = False
MODEL_SAVE_FREQUENCY = 1000
FRAME_RATE = 1./30. # FPS when rending a movie

def main():
    start_time = time()

    # Extract the features of the environment
    # TODO: This has to be automatable.
    n_states = GRID[0] * GRID[1] * GRID[2] * GRID[3]
    n_actions = gym.make(ENV_NAME).action_space.n
    min_values = gym.make(ENV_NAME).observation_space.low
    max_values = gym.make(ENV_NAME).observation_space.high
    # Modify min/max values to something more appropriate for the environment
    min_values[1] = -1.
    min_values[2] = -0.25
    min_values[3] = -2.0
    max_values[1] = 1.
    max_values[2] = 0.25
    max_values[3] = 2.0
    state_encoder = StateEncoder().fit(GRID, min_values, max_values)

    q = {}
    if START_FROM_MODEL:
        # We create a (truly) greedy policy and load a pre-existing q function
        policy = GreedyPolicy()
        with open("best_q.pkl", "rb") as q_file:
            q = pickle.load(q_file)
    else:
        # If we're training a q function de nuovo, we load an epsilon-soft
        # greedy policy and initialize the q function and counts to all zeros.
        # Since the greedy policy breaks ties randomly, this is essentially a
        # random policy stating out
        policy = GreedyPolicy(
            epsilon=EPSILON,
            random_policy=RandomPolicy(n_actions),
        )
        # TODO: I don't think this is a good idea.  Defeats the point of using a
        #       hash tables, and many of the states won't be visitable.  In
        #       addition, I have a feeling that the initial value shouldn't be a
        #       fixed number but rather evolve over time as more information is
        #       known.
        for state in range(n_states):
            q[state] = {key: INITIAL_VALUE for key in range(n_actions)}
        control = SarsaControl(ALPHA, GAMMA, q=q)

    # Outer loop for episode
    for episode in range(NUM_EPSIODES):
        # Create the initial environment and register the initial state
        env, env_raw, observation = make_envs(
            ENV_NAME,
            output_movie=OUTPUT_MOVIE,
            max_steps_per_episode=MAX_STEPS_PER_EPISODE,
        )
        state = state_encoder.transform(observation)

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
            # Update the q function based on the trajector for this episode
            if timestep > 0:
                control.update(trajectory, rewards)
                q = control.get_q()
            observation, reward, done, _ = env.step(action)
            rewards.append(reward)
            state = state_encoder.transform(observation)
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

        close_envs(env, env_raw)
        if episode % MODEL_SAVE_FREQUENCY == 0:
            print("Outputting model...")
            with open("best_q.pkl", "wb") as q_file:
                pickle.dump(q, q_file)
            #print(q)
            #sleep(2)

if __name__ == "__main__":
    main()
