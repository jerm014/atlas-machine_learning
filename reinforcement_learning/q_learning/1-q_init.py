#!/usr/bin/env python3
"""Init the Q-table"""
import numpy as np


def q_init(env):
    """
    Initializes a Q-table for the given environment.

    Args:
       env: A FrozenLakeEnv instance from gymnasium.
           The environment for which to initialize the Q-table.

    Returns:
       numpy.ndarray: A Q-table as a numpy array of zeros with shape
           (number of states, number of actions).

    Note:
       The Q-table is initialized with zeros for all state-action pairs.
       The shape is determined by the observation and action spaces of
       the environment.
    """
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    q_table = np.zeros((n_states, n_actions))
   
    return q_table
