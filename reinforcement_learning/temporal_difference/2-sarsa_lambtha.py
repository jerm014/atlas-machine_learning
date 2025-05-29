#!/usr/bin/env python3
""" This module creates the sarsa_lambtha function"""

import numpy as np
import gym


def policy(state, Q, epsilon):
    """
    Epsilon Greedy Policy

    Returns: action
    """
    if np.random.uniform() < epsilon:
        return np.random.randint(Q.shape[1])
    return np.argmax(Q[state])


def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100, alpha=0.1,
                  gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """
    Performs the SARSA(λ)

    Inputs:\\
    env: openAI environment instance\\
    Q: numpy.ndarray of shape (s,a) containing the Q table\\
    lambtha: eligibility trace factor\\
    episodes: total number of episodes to train over\\
    max_steps: maximum number of steps per episode\\
    alpha: learning rate\\
    gamma: discount rate\\
    epsilon: initial threshold for epsilon greedy\\
    min_epsilon: minimum value that epsilon should decay to\\
    epsilon_decay: decay rate for updating epsilon between episodes

    Returns:\\
    Q: the updated Q table
    """

    # Create empty numpy array of zeros in shape of value estimate
    elig_trace = np.zeros_like(Q)
    # env.seed(0)

    # Loop through episodes
    for ep in range(0, episodes):

        # Reset the environment for each new episode
        state = env.reset()

        # Find action with epsilond_greedy()
        action = policy(state, Q, epsilon)

        # epsilon = max(min_epsilon, epsilon - epsilon_decay)
        epsilon = min_epsilon + (epsilon - min_epsilon) * \
            np.exp(-epsilon_decay * ep)

        # Loop through steps until done or max steps
        for step in range(0, max_steps):

            next_state, reward, done, _ = env.step(action)
            next_action = policy(next_state, Q, epsilon)

            # Temporal Difference Error
            # δ = r + γV(s') - V(s)

            delta = reward + (gamma * Q[next_state][next_action])\
                - Q[state][action]

            # Update eligibility trace and move to the next state
            # Getting closer reults by switching state first
            elig_trace[state][action] = 1.0
            elig_trace = elig_trace * (gamma * lambtha)

            # Update Value Estimate
            Q += delta * alpha * elig_trace
            # Getting worse results with next one
            # V[state] += delta * alpha * elig_trace[state]

            if done:
                break

            # Move state forward
            state = next_state
            action = next_action

    return Q
