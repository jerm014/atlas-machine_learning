#!/usr/bin/env python3
"""Module for Markov chain steady state calculations."""
import numpy as np


def regular(P):
    """
    Calculate steady state probabilities of a regular Markov chain.

    Args:
        P: Square 2D numpy.ndarray transition matrix of shape (n, n)

    Returns:
        numpy.ndarray of shape (1, n) containing steady state probabilities,
        or None on failure
    """
    coditions = [
        isinstance(P, np.ndarray) and len(P.shape) == 2,
        P.shape[0] == P.shape[1],
        np.allclose(np.sum(P, axis=1), 1)]

    if not all(coditions):
        return None

    try:
        eigenvals, eigenvects = np.linalg.eig(P.T)
        close_to_1 = np.isclose(eigenvals, 1)
        if not np.sum(close_to_1) == 1:
            return None
        eigenvect = eigenvects[:, close_to_1].reshape(-1)
        steady_state = eigenvect / np.sum(eigenvect)
        return steady_state.reshape(1, -1)
    except np.linalg.LinAlgError:
        return None
