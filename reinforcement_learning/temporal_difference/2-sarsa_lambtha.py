#!/usr/bin/env python3
"""does basic sarsa"""
import numpy as np


def epsilon_greedy(Q, state, env, epsilon=1):
    """doing epsilon greedy to balance exploration and exploitation """
    p = np.random.uniform()
    if p < epsilon:
        action = np.random.randint(0, env.action_space.n)
        # action = env.action_space.sample()
    else:
        action = np.argmax(Q[state, :])  # following the Q table

    return action


def sarsa_lambtha(
        env,
        Q,
        lambtha,
        num_episodes=5000,
        max_steps=100,
        alpha=0.1,
        gamma=0.99,
        epsilon=1,
        min_epsilon=0.1,
        epsilon_decay=0.05):
    """performs sarsa with gymnasium"""
    E = np.zeros_like(Q)  # shape (64, 4)

    for session in range(num_episodes):
        # Generate one episode
        E.fill(0)  # resetting E
        state, _ = env.reset()
        action = epsilon_greedy(Q, state, env, epsilon)

        for _ in range(max_steps):

            next_state, reward, done, truncated, _ = env.step(action)

            # calculating the next_action on
            next_action = epsilon_greedy(Q, next_state, env, epsilon)
            # the same timestep I guess
            # predict =  Q[state, action]
            # target = reward + (gamma * (Q[next_state, next_action]))
            t_d = reward + gamma * Q[next_state,
                                     next_action] - Q[state, action]
            # I guess env.step(action) always gives r_t+1 ?

            E[state, action] += 1
            Q += alpha * t_d * E

            E *= gamma * lambtha

            state = next_state
            action = next_action

            # epsilon decay? kinda forgot about that but I think it's handled
            # here

            if done:
                break

        # epsilon = max(min_epsilon, (1 - epsilon_decay) * epsilon)

        epsilon = min_epsilon + (1 - min_epsilon) * \
            np.exp(-epsilon_decay * session)

    return Q
