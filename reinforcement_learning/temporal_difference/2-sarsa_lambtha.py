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
        state_info = env.reset()
        # Handle both old and new gym API formats
        if isinstance(state_info, tuple):
            s = state_info[0]
        else:
            s = state_info

        # init eligibility traces, same shape as Q-table
        E = np.zeros_like(Q)
        # choose initial action using epsilon-greedy
        a = choose_action(s, Q, epsilon)

        # loop until episode ends or max steps hit
        for _ in range(max_steps):
            # action, observe next state and reward (and grab term and trunc)
            step_result = env.step(a)

            # Handle different return formats from env.step()
            if len(step_result) == 4:
                # Old gym format: (next_state, reward, done, info)
                next_state, reward, done, _ = step_result
                trunc = False
            else:
                # New gym format: (next_state, reward, term, trunc, info)
                next_state, reward, term, trunc, _ = step_result
                done = term

            # choose next action using epsilon-greedy
            next_action = choose_action(next_state, Q, epsilon)

            # calculate the TD error
            # Key fix: Use accumulating traces, not replacing traces
            if done or trunc:
                delta = reward - Q[s, a]
            else:
                delta = reward + gamma * Q[next_state, next_action] - Q[s, a]

            # Update eligibility trace for current state-action pair FIRST
            E[s, a] += 1.0

            # update Q-table using all eligibility traces
            Q += alpha * delta * E

            # Then decay all eligibility traces
            E *= gamma * lambtha

            # move to the next state,action
            s = next_state
            a = next_action

            # See if episode is done
            if done or trunc:
                break

        # Decay epsilon for the next episode
        epsilon = max(min_epsilon, epsilon - epsilon_decay)

    return Q
