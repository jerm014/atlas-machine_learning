#!/usr/bin/env python3
"""Module for Markov chain probability calculations."""
import numpy as np


def markov_chain(P, s, t=1):
    """Calculate probability of Markov chain states after t iterations.

    Args:
        P: Square 2D numpy.ndarray transition matrix of shape (n, n)
        s: numpy.ndarray initial state probability vector of shape (1, n)
        t: Number of iterations (default=1)

    Returns:
        numpy.ndarray of shape (1, n) representing final state probabilities,
        or None on failure
    """
    if not isinstance(P, np.ndarray) or not isinstance(s, np.ndarray):
        return None
    if len(P.shape) != 2 or P.shape[0] != P.shape[1]:
        return None
    if s.shape[0] != 1 or s.shape[1] != P.shape[0]:
        return None
    if not np.allclose(np.sum(P, axis=1), 1):
        return None
    if not np.allclose(np.sum(s), 1):
        return None

    try:
        return np.matmul(s, np.linalg.matrix_power(P, t))
    except np.linalg.LinAlgError:
        return None
