#!/usr/bin/env python3
"""Implement the epsilon-greedy policy"""
import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """
    Uses epsilon-greedy policy to determine the next action.

    Args:
        Q (numpy.ndarray): Q-table containing action values for each state.
        state (int): The current state index.
        epsilon (float): Exploration rate between 0 and 1.
            Higher values encourage more exploration.

    Returns:
        int: The selected action index.

    Note:
        With probability epsilon, a random action is selected (exploration).
        With probability (1-epsilon), the action with the highest Q-value
        for the current state is selected (exploitation).
    """
    p = np.random.uniform(0, 1)

    # Number of possible actions from the current state
    n_actions = Q.shape[1]

    if p < epsilon:
        # Explore: choose a random action
        action = np.random.randint(0, n_actions)
    else:
        # Exploit: choose the action with the highest Q-value
        action = np.argmax(Q[state])

    return action
