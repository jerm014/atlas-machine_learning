#!/usr/bin/env python3
"""SARSA(λ) Function"""

import numpy as np


def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100, alpha=0.1,
                  gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """Performs SARSA(λ):
        env is the openAI environment instance.
        Q is a numpy.ndarray of shape (s,a) containing the Q table.
        lambtha is the eligibility trace factor.
        episodes is the total number of episodes to train over.
        max_steps is the maximum number of steps per episode.
        alpha is the learning rate.
        gamma is the discount rate.
        epsilon is the initial threshold for epsilon greedy.
        min_epsilon is the minimum value that epsilon should decay to.
        epsilon_decay is the decay rate for updating epsilon between episodes.
        Returns: Q, the updated Q table."""
    n_actions = env.action_space.n
    for _ in range(episodes):
        state = env.reset()
        action = epsilon_greedy(Q, state, n_actions, epsilon)
        eligibility_trace = np.zeros_like(Q)
        for _ in range(max_steps):
            new_state, reward, done, _ = env.step(action)
            new_action = epsilon_greedy(Q, new_state, n_actions, epsilon)
            delta = reward + gamma * Q[new_state, new_action] * (not done) - Q[state, action]
            eligibility_trace[state, action] += 1.0
            Q += alpha * delta * eligibility_trace
            eligibility_trace *= gamma * lambtha
            if done:
                break
            state, action = new_state, new_action
        epsilon = max(epsilon - epsilon_decay, min_epsilon)
    return Q
    
def epsilon_greedy(Q, state, n_actions, epsilon):
    if np.random.rand() < epsilon:
        action = np.random.randint(n_actions)
    else:
        action = np.argmax(Q[state])
    return action
