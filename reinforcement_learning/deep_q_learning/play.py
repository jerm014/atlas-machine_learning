#!/usr/bin/env python3
"""
use a trained DQN agent to play Atari's Breakout
"""
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStack
import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D
from keras.optimizers.legacy import Adam  # Use legacy optimizer
import time  # For adding delay to visualize better

def preprocess_state(state):
    """Convert state from (4, 84, 84) to (84, 84, 4) format"""
    return np.transpose(state, (1, 2, 0))


class DQNAgent:
    def __init__(self, state_shape, action_size):
        self.state_shape = state_shape
        self.action_size = action_size
        self.learning_rate = 0.00025
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Conv2D(32, (8, 8), strides=4, activation='relu',
                         input_shape=self.state_shape))
        model.add(Conv2D(64, (4, 4), strides=2, activation='relu'))
        model.add(Conv2D(64, (3, 3), strides=1, activation='relu'))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def act(self, state):
        # Always exploit, no exploration
        act_values = self.model.predict(state, verbose=0)
        print(f"Action values: {act_values[0]}")  # Debug info
        return np.argmax(act_values[0])

    def load(self, name):
        self.model.load_weights(name)


def main():
    print("Starting Breakout gameplay using trained DQN model...")

    # Create environment for playing 
    env = gym.make('ALE/Breakout-v5', render_mode='human', frameskip=1)
    env = AtariPreprocessing(env, frame_skip=4, grayscale_obs=True,
                           scale_obs=True, terminal_on_life_loss=False)
    env = FrameStack(env, 4)

    # Define state and action space
    state_shape = (84, 84, 4)  # Channels last format for Keras
    action_size = env.action_space.n
    print(f"Action space size: {action_size}")
    print(f"Action meanings: {env.unwrapped.get_action_meanings()}")

    # Create agent for playing
    agent = DQNAgent(state_shape, action_size)

    # Load trained weights
    try:
        agent.load("policy.h5")
        print("Loaded model weights from policy.h5")
    except Exception as e:
        print(f"Error loading model weights: {e}")
        print("Make sure policy.h5 exists. If not, run train.py first.")
        return

    # Play episodes
    episodes = 5
    for e in range(episodes):
        state, _ = env.reset()
        total_reward = 0

        done = False
        steps = 0
        while not done:
            # Convert state shape for the model
            model_input = np.expand_dims(preprocess_state(state), axis=0)

            # Select action
            action = agent.act(model_input)
            print(f"Step {steps}, " +
                  f"Selected action: {action} " +
                  f"({env.unwrapped.get_action_meanings()[action]})")

            # Take action
            next_state, reward, terminated, truncated, _ = env.step(action)

            state = next_state
            total_reward += reward
            done = terminated or truncated

            # Add small delay to see what's happening
            time.sleep(0.05)
            steps += 1

        print(f"Episode: {e+1}/{episodes}, Final Score: {total_reward}")


if __name__ == "__main__":
    main()
