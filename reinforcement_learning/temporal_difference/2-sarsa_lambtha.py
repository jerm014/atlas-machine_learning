#!/usr/bin/env python3
"""Task 2"""

import numpy as np


def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100, alpha=0.1,
                  gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """
    Performs the SARSA(λ) algorithm.

    Parameters:
    env: The OpenAI environment instance.
    Q: A numpy.ndarray of shape (s,a) containing the Q table.
    lambtha: The eligibility trace factor.
    episodes: The total number of episodes to train over.
    max_steps: The maximum number of steps per episode.
    alpha: The learning rate.
    gamma: The discount rate.
    epsilon: The initial threshold for epsilon greedy.
    min_epsilon: The minimum value that epsilon should decay to.
    epsilon_decay: The decay rate for updating epsilon between episodes.

    Returns:
    Q: The updated Q table.
    """
    num_states, num_actions = Q.shape
    E = np.zeros((num_states, num_actions))

    for episode in range(episodes):
        state = env.reset()
        action = np.argmax(Q[state] + np.random.randn(
            1, num_actions) / (episode / 2 + 1))
        for step in range(max_steps):
            new_state, reward, done, info = env.step(action)
            new_action = np.argmax(Q[new_state] + np.random.randn(
                1, num_actions) / (episode / 2 + 1))
            delta = reward + gamma * Q[
                new_state, new_action] - Q[state, action]
            E[state, action] += 1
            Q += alpha * delta * E
            E *= gamma * lambtha
            if done:
                break
            state, action = new_state, new_action
        epsilon = max(min_epsilon, epsilon * epsilon_decay)

    return Q
