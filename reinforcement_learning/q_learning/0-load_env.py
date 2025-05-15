#!/usr/bin/env python3
"""task0"""
import gymnasium as gym


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """
    Loads the FrozenLake environment from gymnasium with specified parameters.
    
    Args:
        desc (List[List[str]]): A custom description of the map to
            load.
            Each element must be one of:
            'S' (starting point)
            'F' (frozen)
            'H' (hole)
            'G' (goal)
        map_name (str): Name of a pre-made map to load.
            Options include '4x4' and '8x8'.
        is_slippery (bool): Whether the ice is slippery.
            Default is False.

    Returns:
        gym.Env: The FrozenLake environment.

    Note:
        If both desc and map_name are None, a randomly generated 8x8 map will be loaded.
        All the arguments are optional and default to None, None, False.
    """
    
    env = gym.make(
        "FrozenLake-v1",
        desc=desc,
        map_name=map_name,
        is_slippery=is_slippery
    )
    
    return env