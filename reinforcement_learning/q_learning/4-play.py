#!/usr/bin/env python3
"""Has a trained agent play an episode of FrozenLake"""
import numpy as np


def play(env, Q, max_steps=100):
    """
    Has the trained agent play an episode of FrozenLake.

    Args:
        env: A FrozenLakeEnv instance from gymnasium with render_mode="ansi".
        Q (numpy.ndarray): Q-table containing the action values.
        max_steps (int): Maximum number of steps in the episode.
            Default is 100.

    Returns:
        tuple: (total_reward, all_renders)
            total_reward is the total reward for the episode
            all_renders is a list of rendered outputs representing the
            board state at each step

    Stuff You Might Care About:
        The agent always exploits the Q-table (no exploration).
        The environment must be initialized with render_mode="ansi".
        The final state of the environment is displayed after the
        episode concludes.
    """
    state, _ = env.reset()
    total_reward = 0
    done = False
    all_renders = []

    # Get initial render of environment
    render = env.render()
    all_renders.append(render)

    for _ in range(max_steps):
        # Always exploit the Q-table (epsilon = 0)
        action = np.argmax(Q[state])

        # Take action
        next_state, reward, done, _, _ = env.step(action)

        # Get render of environment after action
        render = env.render()
        all_renders.append(render)

        # Update state and reward
        state = next_state
        total_reward += reward

        if done:
            break

    return total_reward, all_renders
