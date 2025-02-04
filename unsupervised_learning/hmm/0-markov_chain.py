#!/usr/bin/env python3
"""Module for Markov chain probability calculations."""
import numpy as np


def markov_chain(P, s, t=1):
    """
    Calculate probability of Markov chain states after t iterations.

    Args:
        P: Square 2D numpy.ndarray transition matrix of shape (n, n)
        s: numpy.ndarray initial state probability vector of shape (1, n)
        t: Number of iterations (default=1)

    Returns:
        numpy.ndarray of shape (1, n) representing final state probabilities,
        or None on failure
    """
    conditions = [
        isinstance(P, np.ndarray) and isinstance(s, np.ndarray),
        len(P.shape) == 2 and P.shape[0] == P.shape[1],
        s.shape[0] == 1 and s.shape[1] == P.shape[0],
        np.allclose(np.sum(P, axis=1), 1),
        np.allclose(np.sum(s), 1)]

    if not all(conditions):
        return None

    try:
        return np.matmul(s, np.linalg.matrix_power(P, t))
    except np.linalg.LinAlgError:
        return None
