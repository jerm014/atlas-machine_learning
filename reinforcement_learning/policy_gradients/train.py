#!/usr/bin/env python3
"""train policy gradient"""
import numpy as np


def train(env, np_episodes, alpha=0.000045, gamma=0.98, show_result=False):
    """no"""
    policy_gradient = __import__('policy_gradient').policy_gradient
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    weights = np.random.random((state_dim, action_dim))

    scores = []

    for episode in range(np_episodes):
        # reset env
        state, _ = env.reset()

        states, actions = [], []
        rewards, gradients = [], []
        done, trunc = False, False

        while not (done or trunc):
            # Get action and gradient
            action, gradient = policy_gradient(state, weights)

            next_state, reward, done, truncated, _ = env.step(action)

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            gradients.append(gradient)

            state = next_state

        returns = []
        G = 0
        for reward in reversed(rewards):
            G = reward + gamma * G
            returns.insert(0, G)

        returns = np.array(returns)

        for i in range(len(gradients)):
            weights += alpha * gradients[i] * returns[i]

        score = sum(rewards)
        scores.append(score)
        print(f"Episode: {episode} Score: {score}")

    return scores
