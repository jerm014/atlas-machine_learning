#!/usr/bin/env python3
"""Implements Q-learning training for FrozenLake environment"""
import numpy as np


def train(env,
          Q,
          episodes=5000,
          max_steps=100,
          alpha=0.1,
          gamma=0.99,
          epsilon=1,
          min_epsilon=0.1,
          epsilon_decay=0.05):
    """
    Performs Q-learning on the FrozenLake environment.

    Args:
       env: A FrozenLakeEnv instance from gymnasium.
       Q (numpy.ndarray): Initial Q-table to be updated during training.
       episodes (int): Total number of episodes to train over.
           Default is 5000.
       max_steps (int): Maximum number of steps per episode.
           Default is 100.
       alpha (float): Learning rate. Default is 0.1.
       gamma (float): Discount rate. Default is 0.99.
       epsilon (float): Initial threshold for epsilon greedy.
           Default is 1.
       min_epsilon (float): Minimum value that epsilon should decay to.
           Default is 0.1.
       epsilon_decay (float): Decay rate for updating epsilon between
           episodes. Default is 0.05.
   
    Returns:
       tuple: (Q, total_rewards)
           Q is the updated Q-table
           total_rewards is a list containing the rewards per episode
   
    Note:
       When the agent falls in a hole, the reward is updated to be -1.
    """
    epsilon_greedy = __import__('2-epsilon_greedy').epsilon_greedy
    total_rewards = []

    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0

        for _ in range(max_steps):
            action = epsilon_greedy(Q, state, epsilon)
            next_state, reward, done, _, _ = env.step(action)

            # Apply penalty when falling into a hole
            if done and reward == 0:
                reward = -1

            # Q-learning update equation
            cur_q = Q[state, action]
            max_next_q = np.max(Q[next_state])
            new_q = cur_q + alpha * (reward + gamma * max_next_q - cur_q)
            Q[state, action] = new_q

            state = next_state
            episode_reward += reward

            if done:
                break

        total_rewards.append(episode_reward)

        # Decay epsilon for next episode
        epsilon = max(min_epsilon, epsilon * (1 - epsilon_decay))

    return Q, total_rewards
