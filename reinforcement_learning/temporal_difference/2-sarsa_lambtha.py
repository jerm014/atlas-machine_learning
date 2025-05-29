#!/usr/bin/env python3

"""This might be the only thing I find a how-to on
And I don't actually think it's a how to.
Well Idk which how to that one I was thinking was
But here's a different one:
https://github.com/moripiri/
Reinforcement-Learning-on-FrozenLake/blob/master/Chapter5.ipynb"""

import numpy as np


def sarsa_lambtha(
    env,
    Q,
    lambtha,
    episodes=5000,
    max_steps=100,
    alpha=0.1,
    gamma=0.99,
    epsilon=1,
    min_epsilon=0.1,
    epsilon_decay=0.05,
):
    """env - the environment
    Q - numpy aray of shape (s,a) containing the Q table
    This is the first time we get a Q table in this assignment
    lambtha - eligibility trace factor
    episodes - total epochs/episodes/iterations/whatever
    max_steps - steps before we mercy kill boiyo
    alpha - learning rate
    gamma - discount rate
    epsilon - initial threshold for epsilon greedy
    min_epsilon - value of garunteed random
    epsilon decay - how fast we'll let epsilon decay
    returns an updated q table."""

    eligibility_traces = np.zeros_like(Q)

    for episode in range(episodes):
        state, _ = env.reset()

        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state])

        eligibility_traces.fill(0)

        for step in range(max_steps):
            next_state, reward, terminated, truncated, _ = env.step(action)

            if np.random.rand() < epsilon:
                next_action = env.action_space.sample()
            else:
                next_action = np.argmax(Q[next_state])

            td_error = reward + gamma * Q[
                next_state][next_action] - Q[state][action]

            eligibility_traces[state][action] += 1

            Q += alpha * td_error * eligibility_traces
            eligibility_traces *= gamma * lambtha

            state, action = next_state, next_action

            if terminated or truncated:
                break

        epsilon = max(min_epsilon, epsilon * (1 - epsilon_decay))

    return Q
