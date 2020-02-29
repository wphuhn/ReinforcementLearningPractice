import copy

def update_q_function(q_function_prior, state, action, alpha, reward_action):
    q_function = copy.deepcopy(q_function_prior)
    q_function[state][action] = q_function[state][action] + alpha * (
        reward_action - q_function[state][action]
    )
    return q_function
