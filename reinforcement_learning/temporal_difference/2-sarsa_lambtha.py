#!/usr/bin/env python3
"""
SARSA(λ) algorithm (with eligibility traces), corrected version.
"""
import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """
    Determines the next action using the epsilon-greedy policy.

    Parameters:
        Q (numpy.ndarray): The Q-table, where each entry Q[s, a] represents
                           the expected reward for state `s` and action `a`.
        state (int): The current state.
        epsilon (float): The epsilon value for the epsilon-greedy policy.

    Returns:
        int: The index of the action to take next.
    """
    # Random exploration vs. greedy exploitation
    if np.random.uniform(0, 1) > epsilon:
        return np.argmax(Q[state])
    else:
        return np.random.randint(0, Q.shape[1])


def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100,
                  alpha=0.1, gamma=0.99, epsilon=1.0,
                  min_epsilon=0.1, epsilon_decay=0.05):
    """
    Performs the SARSA(λ) algorithm (with eligibility traces) to estimate
    a Q-table.

    Parameters:
        env (gym.Env): The environment instance (e.g., FrozenLake).
        Q (np.ndarray): The Q-table to update.
        lambtha (float): The eligibility trace factor (λ).
        episodes (int): The total number of episodes to train over.
        max_steps (int): The maximum number of steps per episode.
        alpha (float): The learning rate.
        gamma (float): The discount rate.
        epsilon (float): The initial threshold for epsilon-greedy.
        min_epsilon (float): The lower bound for epsilon.
        epsilon_decay (float): The decay rate for epsilon per episode.

    Returns:
        np.ndarray: The updated Q-table after training.
    """
    initial_epsilon = epsilon

    for episode in range(episodes):
        # Reset environment and get initial state
        state = env.reset()[0]  # for newer gym versions, reset() returns (obs, info)
        action = epsilon_greedy(Q, state, epsilon)

        # Initialize eligibility traces to zero
        eligibility_traces = np.zeros_like(Q)

        for _ in range(max_steps):
            # Interact with the environment
            next_state, reward, done, truncated, _ = env.step(action)
            next_action = epsilon_greedy(Q, next_state, epsilon)

            # Compute TD error
            delta = reward + gamma * Q[next_state, next_action] - Q[state, action]

            # Decay all traces
            eligibility_traces *= gamma * lambtha

            # Increase trace for the visited (state, action)
            eligibility_traces[state, action] += 1

            # Update Q-values
            Q += alpha * delta * eligibility_traces

            # Move to the next state/action
            state, action = next_state, next_action

            if done or truncated:
                break

        # Exponential decay of epsilon after each episode
        epsilon = min_epsilon + (initial_epsilon - min_epsilon) * \
                  np.exp(-epsilon_decay * episode)

    return Q
