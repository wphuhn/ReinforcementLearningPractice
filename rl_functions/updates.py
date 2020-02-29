"""Subroutines for updating non-class objects.

Currently, only the q function update is supported.

    Typical usage example:

    N/A
"""

import copy

def update_q_function(q, state, action, alpha, reward):
    """Update the q function based on reward obtained from a state-action pair.

    The update method used is a simple iterative approach, with scaling factor
    alpha.  If the state exists in the q-function but the state-action pair does
    not, the state-action pair will be initialized to the maximum q value for
    that state before applying the update.  If the state does not exist in the
    q-function, the state-action pair will be initialized to a value of zero
    before applying the update.

    Args:
        q: The q function represented as a two-level nested Iterable, i.e. of
            form q[state][action].  This argument is not updated in-place.
        state: The state in the state-action pair whose q value is updated.
        action: The action in the state-action pair whose q value is updated.
        alpha: The scaling factor for the update.
        reward: The reward for the state-action pair whose q value is updated.

    Returns:
        The updated q function in an identical data format to q.

    Raises:
        None
    """
    q_new = copy.deepcopy(q)
    # if state does not exists in q, we insert it and the action into the
    # dictionary with a zero value
    if state not in q_new:
       q_new[state] = {}
       q_new[state][action] = 0.
    # if state exists in q but the action does not exist, we insert it into the
    # dictionary with the optimistic value
    if action not in q_new[state]:
        max_value = max(q_new[state].values())
        q_new[state][action] = max_value
    q_new[state][action] = q_new[state][action] + alpha * (
        reward - q_new[state][action]
    )
    return q_new
