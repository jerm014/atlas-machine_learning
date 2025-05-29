#!/usr/bin/env python3
"""
SARSA(λ) algorithm
"""
import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """
    Determines the next action using the epsilon-greedy policy.
    """
    # Random exploration vs. greedy exploitation
    if np.random.uniform(0, 1) > epsilon:
        return np.argmax(Q[state])
    else:
        return np.random.randint(0, Q.shape[1])


def sarsa_lambtha(env,
                  Q,
                  lambtha,
                  episodes=5000,
                  max_steps=100,
                  alpha=0.1,
                  gamma=0.99,
                  epsilon=1.0,
                  min_epsilon=0.1,
                  epsilon_decay=0.05
                  ):
    """
    Performs the SARSA(λ) algorithm with eligibility traces to estimate a
    Q-table.
    """
    initial_epsilon = epsilon

    for episode in range(episodes):
        # Reset environment and get initial state
        # for newer gym versions, reset() returns(obs, info)
        state = env.reset()[0]
        action = epsilon_greedy(Q, state, epsilon)

        # Initialize eligibility traces to zero
        eligibility_traces = np.zeros_like(Q)

        for _ in range(max_steps):
            # Interact with the environment
            next_state, reward, done, truncated, _ = env.step(action)
            next_action = epsilon_greedy(Q, next_state, epsilon)

            # Compute TD error
            delta = reward + gamma * Q[next_state, next_action] - \
                Q[state, action]

            # Decay all traces
            eligibility_traces *= gamma * lambtha

            # Increase trace for the visited (state, action)
            eligibility_traces[state, action] += 1

            # update Q-values
            Q += alpha * delta * eligibility_traces

            # move to the next state and action
            state, action = next_state, next_action

            if done or truncated:
                break

        # exponentially decay epsilon after each episode
        epsilon = min_epsilon + (initial_epsilon - min_epsilon) * \
                  np.exp(-epsilon_decay * episode)

    return Q
