#!/usr/bin/env python3
"""implement TD lambda algorithm for task 1"""
import numpy as np


def td_lambtha(env, V, policy, lambtha, episodes=5000, max_steps=100,
               alpha=0.1, gamma=0.99):
    """
    Performs the TD(lambda) algorithm.

    Args:
        env:       environment
        V:         value estimate (it's a numpy.ndarray)
        policy:    function to get the next action
        lambtha:   eligibility trace factor: We don't call this lambda
                   because it would confuse python
        episodes:  total episodes to train on
        max_steps: Maximum steps per episode
        alpha:     learning rate
        gamma:     discount rate

    Returns:
        V:         the updated value estimate
    """
    # Train for a bunch of episodes
    for _ in range(episodes):
        # Reset env and grab initial state
        state, _ = env.reset()

        # Init eligibility traces for all states
        E = np.zeros_like(V)

        # Loop until episode ends or max steps hit
        for _ in range(max_steps):
            # Choose action based on policy
            action = policy(state)

            # action, observe next state and reward (and grab term and trunc)
            next_state, reward, term, trunc, _ = env.step(action)

            # calculate the TD error
            delta = reward + gamma * V[next_state] - V[state]

            # bump the eligibility trace for current state
            E[state] += 1.0

            # update value function and eligibility traces
            V += alpha * delta * E

            # decay eligibility traces
            E *= gamma * lambtha

            # next state
            state = next_state

            # quit when episode is done
            if term or trunc:
                break

    return V
