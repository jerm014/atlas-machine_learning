#!/usr/bin/env python3
"""Imports"""
import numpy as np


def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """that performs SARSA(λ)"""
    for _ in range(episodes):
        state, _ = env.reset()
        epsilon = max(min_epsilon, epsilon - epsilon_decay)
        action = epsilon_greedy(Q, state, epsilon, env.action_space.n)

        eligibility_trace = np.zeros_like(Q)

        for _ in range(max_steps):
            next_state, reward, done, _, _ = env.step(action)
            next_action = epsilon_greedy(Q, next_state, epsilon, env.action_space.n)

            delta = reward + gamma * Q[next_state, next_action] - Q[state, action]
            eligibility_trace *= lambtha * gamma
            eligibility_trace[state, action] += 1

            Q += alpha * delta * eligibility_trace

            state, action = next_state, next_action

            if done:
                break
    return Q


def epsilon_greedy(Q, state, epsilon, n_actions):
    """Epsilon"""
    if np.random.rand() < epsilon:
        return np.random.choice(n_actions)
    return np.argmax(Q[state])
