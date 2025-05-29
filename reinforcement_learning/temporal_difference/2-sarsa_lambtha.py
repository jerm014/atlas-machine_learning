#!/usr/bin/env python3
"""
A module for implementing SARSA(lambda) algorithm
2-sarsa_lambtha.py
"""
import numpy as np


def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100, alpha=0.1,
                  gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """
    Performs SARSA(lambda) for control.

    Args:
        env: environment instance
        Q: Q-table (numpy.ndarray of shape (s,a))
        lambtha: eligibility trace factor (the 'lambda' parameter)
        episodes: total episodes to train over
        max_steps: max steps per episode
        alpha: learning rate
        gamma: discount rate
        epsilon: initial threshold for epsilon-greedy
        min_epsilon: minimum epsilon value
        epsilon_decay: decay rate for epsilon

    Returns:
        Q: the updated Q-table
    """
    def choose_action(state, Q_table, eps):
        """
        Choose action using epsilon-greedy policy.

        Args:
            state: current state
            Q_table: Q-table
            eps: epsilon value

        Returns:
            action: selected action
        """
        if np.random.uniform(0, 1) < eps:
            return np.random.randint(Q_table.shape[1])
        else:
            return np.argmax(Q_table[state, :])

    for episode in range(episodes):
        # Reset environment and eligibility traces
        state, _ = env.reset()
        E = np.zeros_like(Q)
        
        # Choose initial action
        action = choose_action(state, Q, epsilon)

        for step in range(max_steps):
            # Take action and observe result
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # FrozenLake specific: modify reward structure
            if done and reward == 0:
                reward = -1  # Penalty for falling in hole

            # Choose next action BEFORE calculating delta
            next_action = choose_action(next_state, Q, epsilon)

            # Calculate TD error - use next_action for SARSA
            if done:
                delta = reward - Q[state, action]
            else:
                delta = reward + gamma * Q[next_state, next_action] - Q[state, action]

            # CRITICAL DIFFERENCE: Update eligibility first, THEN decay
            E[state, action] += 1.0
            
            # Update ALL Q-values using eligibility traces
            Q += alpha * delta * E
            
            # THEN decay eligibility traces
            E *= gamma * lambtha

            # Move to next state and action
            state = next_state
            action = next_action

            if done:
                break

        # Decay epsilon - try multiplicative decay instead of linear
        epsilon = max(min_epsilon, epsilon * (1 - epsilon_decay))

    return Q