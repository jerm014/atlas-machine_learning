#!/usr/bin/env python3
"""SARSA LAMBTHA!!"""
import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """Espislon Greedy fctn"""
    if np.random.uniform(0, 1) < epsilon:
        return np.random.randint(Q.shape[1])
    else:
        return np.argmax(Q[state])


def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """Perform the SARSA(λ) algorithm for Q table update."""
    for episode in range(episodes):
        state = env.reset()
        state = state[0] if isinstance(state, tuple) else state
        action = epsilon_greedy(Q, state, epsilon)

        eligibility_trace = np.zeros(Q.shape)

        for _ in range(max_steps):
            result = env.step(action)
            next_state, reward, done, *_ = result  # Adjusted unpacking
            next_state = next_state[0] if isinstance(next_state, tuple) else next_state
            next_action = epsilon_greedy(Q, next_state, epsilon)

            delta = reward + gamma * Q[next_state][next_action] - Q[state][action]
            eligibility_trace[state][action] += 1

            Q += alpha * delta * eligibility_trace
            eligibility_trace *= gamma * lambtha

            if done:
                break

            state, action = next_state, next_action

        epsilon = max(min_epsilon, epsilon * (1 - epsilon_decay))

    return Q