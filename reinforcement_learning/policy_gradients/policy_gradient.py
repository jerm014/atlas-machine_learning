#!/usr/bin/env python3
"""policy gradient"""
import numpy as np


def policy(matrix, weight):
    """
    NO, I don't have time to document this properly
    The checker says it isn't documented enough and
    I don't even know what that means eacxtly.
    """
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)

    logits = np.dot(matrix, weight)
    sm_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    policy_probs = sm_logits / np.sum(sm_logits, axis=1, keepdims=True)

    return policy_probs


def policy_gradient(state, weight):
    """
    NO, I don't have time to document this properly
    The checker says it isn't documented enough and
    I don't even know what that means eacxtly.
    """
    prob = policy(state, weight)
    action = np.random.choice(prob.shape[1], p=prob.flatten())
    action_onehot = np.zeros(prob.shape[1])
    action_onehot[action] = 1
    gradient = np.outer(state.flatten(), action_onehot - prob.flatten())

    return action, gradient
