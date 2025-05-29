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
        env:           environment instance
        Q:             Q-table (numpy.ndarray of shape (s,a))
        lambtha:       eligibility trace factor (the 'lambda' parameter)
        episodes:      total episodes to train over
        max_steps:     max steps per episode
        alpha:         learning rate
        gamma:         discount rate
        epsilon:       initial threshold for epsilon-greedy
        min_epsilon:   minimum epsilon value
        epsilon_decay: decay rate for epsilon
    Returns:
        Q:             the updated Q-table
    """
    # helper to choose action with epsilon-greedy
    def choose_action(s, Q, epsilon):
        if np.random.uniform(0, 1) < epsilon:
            # explore
            return np.random.randint(Q.shape[1])
        else:
            # exploit: choose best action, break ties randomly
            return np.argmax(Q[s, :])

    # train for a bunch of episodes
    for _ in range(episodes):
        # reset env and grab initial state
        # Gymnasium returns (observation, info) tuple
        s, _ = env.reset()

        # init eligibility traces, same shape as Q-table
        E = np.zeros_like(Q)
        # choose initial action using epsilon-greedy
        a = choose_action(s, Q, epsilon)

        # loop until episode ends or max steps hit
        for _ in range(max_steps):
            # Take action and observe result
            # Gymnasium returns (observation, reward, terminated, truncated, info)
            next_state, reward, done, trunc, _ = env.step(a)

            # choose next action using epsilon-greedy
            next_action = choose_action(next_state, Q, epsilon)

            # Calculate the TD error - CORRECTED VERSION
            # SARSA update: Q(s,a) <- Q(s,a) + α[r + γQ(s',a') - Q(s,a)]
            if done or trunc:
                # Terminal state: no future value
                delta = reward - Q[s, a]
            else:
                # Non-terminal: use next state-action value
                delta = reward + gamma * Q[next_state, next_action] - Q[s, a]

            # Update eligibility trace for current state-action pair
            E[s, a] += 1.0

            # Update Q-table using all eligibility traces
            Q += alpha * delta * E

            # Decay all eligibility traces AFTER the update
            E *= gamma * lambtha

            # Move to the next state,action
            s = next_state
            a = next_action

            # See if episode is done
            if done or trunc:
                break

        # Decay epsilon for the next episode
        epsilon = max(min_epsilon, epsilon - epsilon_decay)

    return Q
