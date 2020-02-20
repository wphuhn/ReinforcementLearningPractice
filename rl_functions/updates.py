import copy

def update_q_function(q_function_prior, action, alpha, reward_action):
    q_function = copy.deepcopy(q_function_prior)
    q_function[action] = q_function[action] + alpha * (reward_action - q_function[action])
    return q_function
