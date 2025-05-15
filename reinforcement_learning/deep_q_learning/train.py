#!/usr/bin/env python3
"""
training script for DQN to play Atari's Breakout
"""
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStack
import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D
from keras.optimizers.legacy import Adam
from collections import deque
import random
import os
import time

class DQNAgent:
    def __init__(self, state_shape, action_size):
        self.state_shape = state_shape
        self.action_size = action_size
        self.memory = deque(maxlen=50000)  # Increased memory size
        self.gamma = 0.99  # discount rate
        self.epsilon = 1.0  # exploration rate
        # Lower min epsilon for more exploitation eventually
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995  # Slower decay
        self.learning_rate = 0.00025
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

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

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, training=True):
        if training and np.random.rand() <= self.epsilon:
            # Encourage movement during exploration!
            # 70% of exploration should be movement
            if np.random.rand() < 0.7:
                return np.random.choice([2, 3])  # RIGHT or LEFT
            else:
                return np.random.randint(0, self.action_size)

        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)

        states = []
        next_states = []

        for state, _, _, next_state, _ in minibatch:
            processed_state = preprocess_state(state[0])
            processed_next_state = preprocess_state(next_state[0])
            states.append(processed_state)
            next_states.append(processed_next_state)

        states = np.array(states)
        next_states = np.array(next_states)

        targets = self.model.predict(states, verbose=0)
        next_q_values = self.target_model.predict(next_states, verbose=0)

        for i, (_, action, reward, _, done) in enumerate(minibatch):
            if done:
                targets[i, action] = reward
            else:
                gamma_max_n_q_v_i = self.gamma * np.max(next_q_values[i])
                targets[i, action] = reward + gamma_max_n_q_v_i

        self.model.fit(states,
                       targets,
                       epochs=1,
                       verbose=0,
                       batch_size=batch_size)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


def preprocess_state(state):
    """Convert state from (4, 84, 84) to (84, 84, 4) format"""
    return np.transpose(state, (1, 2, 0))


def main():
    print("Starting DQN training...")

    # Create environment with visible training
    use_rendering = False  # False for faster training, True to see it!
    render_mode = 'human' if use_rendering else None

    env = gym.make('ALE/Breakout-v5', render_mode=render_mode, frameskip=1)
    env = AtariPreprocessing(env, frame_skip=4, grayscale_obs=True,
                           scale_obs=True, terminal_on_life_loss=True)
    env = FrameStack(env, 4)

    # Get a sample state
    sample_state, _ = env.reset()
    print(f"Original state shape: {sample_state.shape}")

    # Define state and action space
    state_shape = (84, 84, 4)
    action_size = env.action_space.n
    print(f"Action space size: {action_size}")
    print(f"Action meanings: {env.unwrapped.get_action_meanings()}")

    # Create agent
    agent = DQNAgent(state_shape, action_size)

    # Load previous weights if exists
    if os.path.exists("policy.h5"):
        try:
            agent.load("policy.h5")
            print("Loaded existing weights, continuing training...")
        except:
            print("Could not load existing weights, starting fresh...")

    # Training parameters
    batch_size = 64
    episodes = 10000
    update_target_every = 500
    save_every = 50

    # Training loop
    steps = 0
    best_reward = -float('inf')

    for e in range(episodes):
        state, _ = env.reset()
        state = np.expand_dims(state, axis=0)
        total_reward = 0

        # Fire to start the game
        action = 1  # FIRE
        next_state, reward, done, truncated, _ = env.step(action)
        next_state = np.expand_dims(next_state, axis=0)

        state = next_state
        # Track actions for debugging
        action_counts = {0: 0, 1: 0, 2: 0, 3: 0}

        for t in range(10000):  # Max steps per episode
            model_input = np.expand_dims(preprocess_state(state[0]), axis=0)

            # Select action
            action = agent.act(model_input)
            action_counts[action] += 1

            # Take action
            next_state, reward, done, truncated, _ = env.step(action)
            next_state = np.expand_dims(next_state, axis=0)

            # Encourage movement to avoid getting stuck
            if action in [2, 3]:  # RIGHT or LEFT
                reward += 0.01  # Small bonus for moving

            # Remember experience
            agent.remember(state,
                           action,
                           reward,
                           next_state,
                           done or truncated)

            # Update state and counters
            state = next_state
            total_reward += reward
            steps += 1

            # Train the network
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

            # Update target network
            if steps % update_target_every == 0:
                agent.update_target_model()
                print(f"Step {steps}: Updated target " +
                      f"model, epsilon: {agent.epsilon:.4f}")

            if done or truncated:
                break

            # Brief delay if rendering to allow viewing
            if use_rendering:
                time.sleep(0.01)

        print(f"Episode: {e+1}/{episodes}, " +
              f"Score: {total_reward}, " +
              f"Epsilon: {agent.epsilon:.4f}")
        print(f"Action counts: {action_counts}")

        # Save if it's the best model so far
        if total_reward > best_reward:
            best_reward = total_reward
            agent.save("best_policy.h5")
            print(f"New best reward: {best_reward}, " +
                  "model saved to best_policy.h5")

        # Save periodically
        if (e + 1) % save_every == 0:
            agent.save("policy.h5")
            print(f"Saved model at episode {e+1}")

    # Save the final model
    agent.save("policy.h5")
    print("Training finished, model saved to policy.h5")

    if os.path.exists("best_policy.h5"):
        print(f"Best model with reward {best_reward} " +
              "saved to best_policy.h5")


if __name__ == "__main__":
    main()