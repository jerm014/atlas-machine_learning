#!/usr/bin/env python3
"""Module for Hidden Markov Model forward and Viterbi algorithm
calculations."""
import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
    """Calculate most likely hidden state sequence using Viterbi algorithm.

    Args:
        Observation: numpy.ndarray of shape (T,) containing observation
                     indices
        Emission:    numpy.ndarray of shape (N, M) containing emission
                     probabilities
        Transition:  numpy.ndarray of shape (N, N) containing transition
                     probs
        Initial:     numpy.ndarray of shape (N, 1) containing starting
                     probabilities

    Returns:  path, P
        path: List of length T containing most likely sequence of hidden
              states
        P:    Probability of obtaining the path sequence
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
        viterbi = np.zeros((N, T))
        backpointer = np.zeros((N, T), dtype=int)
        viterbi[:, 0] = np.log(Initial.flatten() * Emission[:, Observation[0]])

        for t in range(1, T):
            for n in range(N):
                trans_probs = viterbi[:, t-1] + np.log(Transition[:, n])
                backpointer[n, t] = np.argmax(trans_probs)
                viterbi[n, t] = trans_probs[backpointer[n, t]] + \
                    np.log(Emission[n, Observation[t]])

        path = []
        current = np.argmax(viterbi[:, -1])
        path.append(current)

        for t in range(T-1, 0, -1):
            current = backpointer[current, t]
            path.append(current)

        path.reverse()
        P = float(np.exp(viterbi[path[-1], -1]))

        return path, P

    except Exception:
        return None, None
