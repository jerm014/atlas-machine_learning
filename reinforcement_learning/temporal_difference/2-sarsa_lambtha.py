#!/usr/bin/env python3
"""A module for implementing SARSA(lambda) algorithm"""
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
    def choose_action(state, Q, epsilon):
        if np.random.uniform(0, 1) < epsilon:
            # explore
            return env.action_space.sample()
        else:
            # exploit: choose best action, break ties randomly
            return np.argmax(Q[state, :])

    # train for a bunch of episodes
    for _ in range(episodes):
        # reset env and grab initial state
        state, _ = env.reset()
        # init eligibility traces, same shape as Q-table
        E = np.zeros_like(Q)
        # choose initial action using epsilon-greedy
        action = choose_action(state, Q, epsilon)

        # loop until episode ends or max steps hit
        for _ in range(max_steps):
            # action, observe next state and reward (and grab term and trunc)
            next_state, reward, term, trunc, _ = env.step(action)

            # choose next action using epsilon-greedy
            next_action = choose_action(next_state, Q, epsilon)

            # calculate the TD error
            # if term, next state has no future value (Q[next_state,
            # next_action] effectively 0)
            if term:
                delta = reward - Q[state, action]
            else:
                delta = reward + gamma * Q[next_state, next_action] \
                        - Q[state, action]

            # bump up eligibility trace for current state-action pair
            E[state, action] += 1.0

            # update Q-table and eligibility traces
            Q += alpha * delta * E

            # decay eligibility traces
            E *= gamma * lambtha

            # move to next state, action
            state = next_state
            action = next_action

            # see if episode is done
            if term or trunc:
                break

        # decay epsilon for the next episode
        epsilon = max(min_epsilon, epsilon - epsilon_decay)

    return Q
