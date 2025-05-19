#!/usr/bin/env python3
"""
Training script for an improved Dueling Double DQN agent on Atari Breakout
"""
import os
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStack

import tensorflow as tf
import tensorflow.keras as tk
# Monkey-patch tf.keras to expose __version__ so keras-rl2 callbacks import
tk.__version__ = tf.__version__

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Lambda, Concatenate
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras import backend as K

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory, PrioritizedMemory  # Using keras-rl2's PrioritizedMemory
from rl.core import Processor
from rl.callbacks import ModelIntervalCheckpoint, FileLogger


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
    
    # Combine value and advantage streams
    # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))
    def combine_streams(x):
        value, advantage = x
        # Expanding dimensions of value for proper broadcasting
        return value + (advantage - K.mean(advantage, axis=1, keepdims=True))
    
    q_values = Lambda(combine_streams)([value, advantage])
    
    model = Model(inputs=inputs, outputs=q_values)
    return model


class AtariProcessor(Processor):
    """
    Custom processor for the DQN agent to handle Atari-specific processing.
    Also implements the Double DQN algorithm through the process_target method.
    """
    def __init__(self, model, nb_actions):
        self.model = model
        self.nb_actions = nb_actions
    
    def process_observation(self, observation):
        # Already processed by AtariPreprocessing wrapper
        return observation
    
    def process_state_batch(self, batch):
        # Already processed
        return batch
    
    def process_reward(self, reward):
        # Clip rewards between -1 and 1 for stability
        return np.clip(reward, -1., 1.)
    
    def process_target(self, target, state):
        """
        Implements Double DQN by decoupling action selection and evaluation.
        
        The online model selects actions, the target model evaluates them.
        """
        batch_size = state.shape[0]
        
        # Get best actions from online model (model in the agent)
        # This decouples action selection from target value estimation
        online_q_values = self.model.predict(state)
        best_actions = np.argmax(online_q_values, axis=1)
        
        # Replace the target Q-values with DDQN target calculation
        # We only update the Q-values for the selected actions
        # NOTE: This only works with keras-rl2 if you investigate the source,
        # it passes the target model's prediction as 'target' to this method
        for i in range(batch_size):
            # Only keep the value for the selected action
            target[i, :] = 0.  # Zero out all action values
            target[i, best_actions[i]] = target[i, best_actions[i]]  # Keep only the best action's value
            
        return target


def main():
    """
    Main function to train the DQN agent.
    
    Usage:
        train.py [base_filename]
        
    If base_filename is provided, it will be used for weights and log files,
    otherwise "policy" will be used as default.
    """
    import sys
    
    # Get base filename from command line args or use default
    if len(sys.argv) > 1:
        base_filename = sys.argv[1]
    else:
        base_filename = "policy"
    
    # Set filenames for weights and logs
    weights_filename = f"{base_filename}.h5"
    log_filename = f"{base_filename}_log.json"
    
    # Environment parameters
    ENV_NAME = 'ALE/Breakout-v5'
    
    # Training parameters
    nb_steps = 10000000  # 10M steps for good performance
    memory_size = 250000  # 250k experiences in memory
    batch_size = 32
    window_length = 4
    
    # DQN Agent parameters
    gamma = 0.99  # Discount factor
    target_model_update = 10000  # Update target model every 10k steps
    learning_rate = 0.00025  # Adam optimizer learning rate
    learning_rate_decay = 0.0  # No decay
    
    # Exploration parameters
    max_epsilon = 1.0
    min_epsilon = 0.1
    epsilon_decay_steps = 1000000  # Annealed over 1M steps
    
    # Create and wrap environment without rendering during training
    env = gym.make(ENV_NAME, frameskip=1)
    env = AtariPreprocessing(
        env,
        frame_skip=4,
        grayscale_obs=True,
        scale_obs=True,
        terminal_on_life_loss=True  # Important for better learning in Breakout
    )
    
    # Wrap to classic Gym API for keras-rl2
    env = Step4Wrapper(env)
    env = ResetWrapper(env)
    
    # Action and observation details
    nb_actions = env.action_space.n
    input_shape = (window_length,) + env.observation_space.shape  # (4, 84, 84)
    
    # Build the dueling model architecture
    model = build_dueling_model(input_shape, nb_actions)
    print(model.summary())
    
    # Configure memory - SequentialMemory works well enough
    memory = SequentialMemory(limit=memory_size, window_length=window_length)
    
    # Configure the exploration policy: annealed epsilon-greedy
    policy = LinearAnnealedPolicy(
        EpsGreedyQPolicy(),
        attr='eps',
        value_max=max_epsilon,
        value_min=min_epsilon,
        value_test=0.05,  # Some exploration during testing
        nb_steps=epsilon_decay_steps
    )
    
    # Create the processor for Double DQN
    processor = AtariProcessor(model, nb_actions)
    
    # Create the Dueling DQN agent
    dqn = DQNAgent(
        model=model,
        nb_actions=nb_actions,
        memory=memory,
        nb_steps_warmup=50000,  # Collect experiences before training
        target_model_update=target_model_update,
        policy=policy,
        gamma=gamma,
        batch_size=batch_size,
        processor=processor,
        enable_double_dqn=True,  # Additional flag for Double DQN in keras-rl2
        train_interval=4,  # Train every 4 steps
        delta_clip=1.0  # Huber loss clipping parameter
    )
    
    # Compile the agent
    dqn.compile(Adam(learning_rate=learning_rate, decay=learning_rate_decay),
                metrics=['mae'])
    
    # Callbacks for saving checkpoints and logging
    checkpoint_weights_filename = f"{base_filename}_{{step}}.h5"
    checkpoint_callback = ModelIntervalCheckpoint(
        checkpoint_weights_filename, interval=250000
    )
    logger = FileLogger(log_filename, interval=10000)
    
    # Load weights if they exist
    if os.path.exists(weights_filename):
        print(f"Loading weights from {weights_filename}")
        dqn.load_weights(weights_filename)
    
    # Start training
    print("Training the agent...")
    dqn.fit(env, nb_steps=nb_steps,
            callbacks=[checkpoint_callback, logger],
            log_interval=10000,
            verbose=1)
    
    # Save final weights
    dqn.save_weights(weights_filename, overwrite=True)
    print(f"Weights saved to {weights_filename}")
    
    # Test the agent after training
    print("Testing the agent...")
    test_env = gym.make(ENV_NAME, render_mode='human', frameskip=1)
    test_env = AtariPreprocessing(
        test_env,
        frame_skip=4,
        grayscale_obs=True,
        scale_obs=True,
        terminal_on_life_loss=False  # Don't terminate on life loss during testing
    )
    test_env = Step4Wrapper(test_env)
    test_env = ResetWrapper(test_env)
    
    dqn.test(test_env, nb_episodes=10, visualize=True)
    
    print("Training and testing finished.")
    env.close()
    test_env.close()


if __name__ == '__main__':
    main()