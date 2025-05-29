#!/usr/bin/env python3
"""Monte Carlo"""
import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100, alpha=0.1,
                gamma=0.99):
    """no time for docs tonight"""
    def run_single_ep():
        """no time for docs tonight"""
        state = []
        reward = []
        current_pos, _ = env.reset()

        for _ in range(max_steps):
            move = policy(current_pos)

            new_pos, immediatereward, term, trunc, _ = env.step(move)

            state.append(int(current_pos))
            reward.append(int(immediatereward))

            if term or trunc:
                break

            current_pos = new_pos

        return np.array(state), np.array(reward)

    # learning loop
    for trial_num, _ in enumerate(range(episodes)):
        # generate episode path
        path, rewards_seq = run_single_ep()

        # process traj backwards to calculate returns
        cumulative_return = 0

        # process backwards using negative indexing
        for position in range(len(path)):
            reverse_idx = -(position + 1)
            location = path[reverse_idx]
            immediate_payoff = rewards_seq[reverse_idx]

            # calculate discounted return
            cumulative_return = immediate_payoff + gamma * cumulative_return

            # update if state isn't in first trial_num position
            if trial_num < len(path):
                early_positions = path[:trial_num]
            else:
                early_positions = path[:len(path)]

            # update value function if its new
            if location not in early_positions:
                prediction_error = cumulative_return - V[location]
                V[location] += alpha * prediction_error

    return V
