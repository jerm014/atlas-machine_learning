#!/usr/bin/env python3
"""
Performs the SARSA(λ) algorithm for Q-value estimation.
"""

import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """
    Uses epsilon-greedy to determine the next action.

    Args:
        Q (numpy.ndarray): A numpy.ndarray containing the q-table.
        state (int): The current state.
        epsilon (float): The epsilon value.

    Returns:
        int: The next action index.
    """
    p = np.random.uniform()
    if p < epsilon:
        # Explore: Choose a random action
        # Q.shape[1] is the number of actions (number of columns in Q-table)
        action = np.random.randint(Q.shape[1])
    else:
        # Exploit: Choose the action with the highest Q-value for the current state
        action = np.argmax(Q[state, :])
    return action


def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100, alpha=0.1,
                  gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.001):
    """
    Performs the SARSA(λ) algorithm for Q-value estimation.

    Args:
        env (object): The environment instance (expected to be a Gymnasium Env).
        Q (numpy.ndarray): A numpy.ndarray of shape (s,a) containing the Q-table.
        lambtha (float): The eligibility trace factor.
        episodes (int, optional): The total number of episodes to train over.
            Defaults to 5000.
        max_steps (int, optional): The maximum number of steps per episode.
            Defaults to 100.
        alpha (float, optional): The learning rate. Defaults to 0.1.
        gamma (float, optional): The discount rate. Defaults to 0.99.
        epsilon (float, optional): The initial threshold for epsilon greedy.
            Defaults to 1.
        min_epsilon (float, optional): The minimum value that epsilon should
            decay to. Defaults to 0.1.
        epsilon_decay (float, optional): The decay rate for updating epsilon
            between episodes. Defaults to 0.001.

    Returns:
        numpy.ndarray: Q, the updated Q table.
    """
    for episode in range(episodes):
        # Initialize eligibility traces for the new episode
        # E has the same shape as Q, initialized to zeros
        E = np.zeros_like(Q)

        # Reset environment and get initial state
        # env.reset() returns (observation, info), we only need the observation
        state = env.reset()[0]
        # Choose initial action using epsilon-greedy policy
        action = epsilon_greedy(Q, state, epsilon)

        done = False
        truncated = False

        for step in range(max_steps):
            # Take the chosen action
            new_state, reward, done, truncated, _ = env.step(action)

            # Adjust reward if agent falls into a hole (consistent with previous tasks)
            # In FrozenLake, reward is 0 for holes, but we want -1 for learning
            if done and reward == 0:
                reward = -1

            # Choose the next action from the new state using epsilon-greedy
            # This is the "A'" in SARSA(λ) update
            new_action = epsilon_greedy(Q, new_state, epsilon)

            # Calculate TD error (delta)
            # If new_state is a terminal state, its Q-value for bootstrapping is 0
            if done or truncated:
                td_target = reward
            else:
                td_target = reward + gamma * Q[new_state, new_action]

            td_error = td_target - Q[state, action]

            # Update eligibility trace for the current state-action pair
            # This marks the current state-action pair as "eligible" for update
            E[state, action] += 1

            # Update Q for all state-action pairs and decay eligibility traces
            # The update rule for Q is applied to all state-action pairs based on their eligibility
            # The eligibility traces then decay for the next iteration
            Q += alpha * td_error * E
            E *= gamma * lambtha

            # Move to the next state and action for the next iteration
            state = new_state
            action = new_action

            if done or truncated: # Check both done and truncated
                break

        # Decay epsilon for the next episode (linear decay)
        epsilon = max(min_epsilon, epsilon - epsilon_decay)

    return Q
