#!/usr/bin/env python3
"""
Play script for DQN agent trained on Atari Breakout
"""
import os
import sys
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing

import tensorflow as tf
import tensorflow.keras as tk
# Monkey-patch tf.keras to expose __version__ so keras-rl2 callbacks import
tk.__version__ = tf.__version__

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Lambda
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers.legacy import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import GreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor


class Step4Wrapper(gym.Wrapper):
    """Wrap step() to return classic 4-tuple (obs, reward, done, info)."""
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return obs, reward, done, info


class ResetWrapper(gym.Wrapper):
    """Wrap reset() to return only observation (drop info)."""
    def reset(self, **kwargs):
        obs, _ = self.env.reset(**kwargs)
        return obs


# Define the combine_streams function outside of build_dueling_model
# to make it available globally for model loading
def combine_streams(inputs):
    """Combines value and advantage streams for dueling DQN architecture."""
    value, advantage = inputs
    return value + (advantage - K.mean(advantage, axis=1, keepdims=True))


def build_dueling_model(input_shape, nb_actions):
    """
    Builds the Dueling network architecture for the DQN agent.
    
    Args:
        input_shape (tuple): Shape of the input (window_length, height, width).
        nb_actions (int): Number of possible actions.
    
    Returns:
        A Keras Model with dueling architecture.
    """
    # Input layer
    inputs = Input(shape=input_shape)
    
    # Convolutional layers
    conv1 = Conv2D(32, (8, 8), strides=(4, 4), activation='relu', 
                  data_format='channels_first')(inputs)
    conv2 = Conv2D(64, (4, 4), strides=(2, 2), activation='relu', 
                  data_format='channels_first')(conv1)
    conv3 = Conv2D(64, (3, 3), strides=(1, 1), activation='relu', 
                  data_format='channels_first')(conv2)
    
    flatten = Flatten()(conv3)
    
    # Dueling architecture - split into value and advantage streams
    # Value stream (state value)
    value_stream = Dense(512, activation='relu')(flatten)
    value = Dense(1, activation='linear')(value_stream)
    
    # Advantage stream (action advantages)
    advantage_stream = Dense(512, activation='relu')(flatten)
    advantage = Dense(nb_actions, activation='linear')(advantage_stream)
    
    # Combine value and advantage streams using the globally defined function
    q_values = Lambda(combine_streams)([value, advantage])
    
    model = Model(inputs=inputs, outputs=q_values)
    return model


class AtariProcessor(Processor):
    """Custom processor for the DQN agent to handle Atari-specific processing."""
    def process_observation(self, observation):
        # Already processed by AtariPreprocessing wrapper
        return observation
    
    def process_state_batch(self, batch):
        # Already processed
        return batch
    
    def process_reward(self, reward):
        # No reward clipping during testing
        return reward


def Play(weights_basename, nb_episodes=10):
    """
    Play Breakout using a trained DQN agent.
    
    Args:
        weights_basename (str): Base filename for the weights file (without .h5)
        nb_episodes (int): Number of episodes to play
    """
    weights_filename = f"{weights_basename}.h5"
    ENV_NAME = 'ALE/Breakout-v5'
    window_length = 4
    
    # Create and wrap environment with rendering
    env = gym.make(ENV_NAME, render_mode='human', frameskip=1)
    env = AtariPreprocessing(
        env,
        frame_skip=4,
        grayscale_obs=True,
        scale_obs=True,
        terminal_on_life_loss=False  # Don't terminate on life loss during testing
    )
    
    # Wrap to classic Gym API for keras-rl2
    env = Step4Wrapper(env)
    env = ResetWrapper(env)
    
    # Action and observation details
    nb_actions = env.action_space.n
    input_shape = (window_length,) + env.observation_space.shape  # (4, 84, 84)
    
    # Build the same model architecture used for training
    model = build_dueling_model(input_shape, nb_actions)
    
    # Configure memory - needed for DQNAgent init, but not for playing
    # Small memory is fine as we only test
    memory = SequentialMemory(limit=1000, window_length=window_length)
    
    # Configure policy for playing: GreedyQPolicy
    policy = GreedyQPolicy()
    
    # Create processor for Atari
    processor = AtariProcessor()
    
    # Create DQNAgent
    dqn = DQNAgent(
        model=model,
        nb_actions=nb_actions,
        memory=memory,
        processor=processor,
        nb_steps_warmup=10,  # small value, doesn't matter for test
        target_model_update=10,  # small value, doesn't matter for test
        policy=policy,
        gamma=0.99,  # does not matter for test
        enable_double_dqn=True  # Keep this consistent with training
    )
    
    # Compile the agent. Optimizer/loss don't matter for playing
    dqn.compile(Adam(learning_rate=0.00025), metrics=['mae'])
    
    # Load trained weights
    if os.path.exists(weights_filename):
        try:
            dqn.load_weights(weights_filename)
            print(f"Loaded weights from {weights_filename}")
        except Exception as e:
            print(f"Failed to load weights from {weights_filename}: {e}")
            print("Cannot play without loaded weights.")
            env.close()
            return
    else:
        print(f"Weights file not found at {weights_filename}.")
        print("Cannot play without trained weights.")
        env.close()
        return
    
    # Play the game for the specified number of episodes
    print(f"Playing Breakout for {nb_episodes} episodes...")
    dqn.test(env, nb_episodes=nb_episodes, visualize=True)
    
    print("Playing finished.")
    env.close()
    print("Environment closed.")


def main():
    """Main function when play.py is run directly."""
    if len(sys.argv) > 1:
        weights_basename = sys.argv[1]
    else:
        weights_basename = "policy"  # Default weights filename

    nb_episodes = 10
    if len(sys.argv) > 2:
        try:
            nb_episodes = int(sys.argv[2])
        except ValueError:
            print(f"Invalid number of episodes: {sys.argv[2]}. Using default: {nb_episodes}")
    
    Play(weights_basename, nb_episodes)


if __name__ == '__main__':
    main()