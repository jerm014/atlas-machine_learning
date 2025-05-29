#!/usr/bin/env python3
"""
performs the SARSA(λ) algorithm
epsilon greedy is copied directly from my previous project
    in the q_learning dir
"""

import numpy as np
# epsilon_greedy = __import__('../q_learning/2-epsilon_greedy').epsilon_greedy


def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100,
                  alpha=0.1, gamma=0.99, epsilon=1,
                  min_epsilon=0.1, epsilon_decay=0.05):
    """
    env = openai environment instance
    Q = qtable
        np.ndarray
            shape = (s,a)
                s = number of states
                a = number of actions
    lambtha = eligibility trace factor
    episodes = num episodes for training
    max_steps = steps per episode
    alpha = learning rate
    gamma = discount rate
    epsilon = initial threshold for epsilon greedy
        float
    epsilon_decay = how much to decrease epsilon between episodes
    min_epsilon = epsilon must go below this value

    return Q (updated)
    """
    qtable = Q
    num_states, num_actions = np.shape(qtable)
    for episode in range(episodes):
        # eligibility trace init at 0 for each state
        elig_trace = np.zeros_like(qtable)
        current_state = env.reset()
        policy = epsilon_greedy

        for step_num in range(max_steps):
            # I should make a `step` helper function for clarity
            # this bit is the same as monte carlo
            # policy will just be epsilon greedy
            # we are already provided the things we need to make the funciton
            # we can even import the already made epsilon greedy policy from
            # old projects
            current_action = policy(qtable, current_state, epsilon)
            next_state, reward, terminal, _ = env.step(current_action)
            next_action = policy(qtable, next_state, epsilon)

            # TD Error
            delta = reward + gamma * \
                qtable[next_state, next_action] -\
                qtable[current_state, current_action]
            # eligibility only assigned to sates actually visited
            elig_trace[current_state, current_action] += 1
            qtable += alpha * delta * elig_trace
            elig_trace[current_state, current_action] *= gamma * lambtha

            if terminal:
                break
        # epsilon update between episodes
        epsilon = max(epsilon - epsilon_decay, min_epsilon)
    return qtable


def epsilon_greedy(Q, state, epsilon):
    """
    direct copy of previous work
        to save time learning to import properly
            from adjacent dir
    Q = numpy.ndarray containing q-table
    state = current state
        int
    epsilon = epsilon used for  calculation
    returns action_index
        int
    """
    # clarity
    qtable = Q
    _, act_count = np.shape(qtable)
    state_row = qtable[state]

    greed = np.random.uniform(0.0, 1.0) < epsilon
    if greed:
        return greed_is_good(act_count)
    else:
        return exploiter_orb(state_row)


def greed_is_good(act_count):
    """
    recieves a state
    returns greed route (randomly take a path)
        aka explore
    """
    return np.random.randint(0, act_count)


def exploiter_orb(state_row):
    """
    recieves a state row
        of a qtable
    returns exploit route
        = index of highest known reward
    """
    return np.argmax(state_row)
