#!/usr/bin/env python3
"""Module for Hidden Markov Model forward, backward, and Viterbi algorithms."""
import numpy as np


def backward(Observation, Emission, Transition, Initial):
    """Calculate backward probabilities for Hidden Markov Model.

    Args:
        Observation: numpy.ndarray of shape (T,) containing observation
                     indices
        Emission:    numpy.ndarray of shape (N, M) containing emission
                     probabilities
        Transition:  numpy.ndarray of shape (N, N) containing transition
                     probabilities
        Initial:     numpy.ndarray of shape (N, 1) containing starting
                     probabilities

    Returns:  P, B
        P: Likelihood of observations given the model
        B: numpy.ndarray of shape (N, T) containing backward path
           probabilities
        or None, None on failure
    """

    conditions = [
        isinstance(Observation, np.ndarray),
        isinstance(Emission, np.ndarray),
        isinstance(Transition, np.ndarray),
        isinstance(Initial, np.ndarray)
    ]

    if not all(conditions):
        return None, None

    conditions = [
        len(Transition.shape) == 2,
        len(Initial.shape) == 2,
        len(Emission.shape) == 2,
        Transition.shape[0] == Transition.shape[1],
        Initial.shape[1] == 1,
        Initial.shape[0] == Transition.shape[0],
        np.allclose(np.sum(Transition, axis=1), 1),
        np.allclose(np.sum(Emission, axis=1), 1),
        np.allclose(np.sum(Initial), 1)
    ]

    if not all(conditions):
        return None, None

    try:

        T = Observation.shape[0]
        N = Transition.shape[0]
        B = np.zeros((N, T))
        B[:, -1] = 1

        for t in range(T - 2, -1, -1):
            for n in range(N):
                B[n, t] = np.sum(
                    Transition[n, :] * Emission[:, Observation[t + 1]] *
                    B[:, t + 1]
                )

        P = np.sum(Initial.flatten() * Emission[:, Observation[0]] * B[:, 0])

        return P, B

    except Exception:
        return None, None
