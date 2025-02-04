#!/usr/bin/env python3
"""Module for Hidden Markov Model forward algorithm calculations."""
import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """Calculate forward probabilities for Hidden Markov Model.

    Args:
        Observation: numpy.ndarray of shape (T,) containing observation indices
        Emission:    numpy.ndarray of shape (N, M) containing emission
                     probabilities
        Transition:  numpy.ndarray of shape (N, N) containing transition
                     probabilities
        Initial:     numpy.ndarray of shape (N, 1) containing starting
                     probabilities

    Returns:
        P: Likelihood of observations given the model
        F: numpy.ndarray of shape (N, T) containing forward path probabilities
        or None, None on failure
    """
    try:
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
            len(Emission.shape) == 2
        ]
        if not all(conditions):
            return None, None

        conditions = [
            Transition.shape[0] == Transition.shape[1],
            Initial.shape[1] == 1,
            Initial.shape[0] == Transition.shape[0],
            np.allclose(np.sum(Transition, axis=1), 1),
            np.allclose(np.sum(Emission, axis=1), 1),
            np.allclose(np.sum(Initial), 1)
        ]
        if not all(conditions):
            return None, None

        T = Observation.shape[0]
        N, M = Emission.shape

        F = np.zeros((N, T))
        F[:, 0] = Initial.flatten() * Emission[:, Observation[0]]

        for t in range(1, T):
            for n in range(N):
                F[n, t] = np.sum(F[:, t-1] * Transition[:, n]) * \
                            Emission[n, Observation[t]]

        P = np.sum(F[:, -1])
        return P, F

    except Exception:
        return None, None
