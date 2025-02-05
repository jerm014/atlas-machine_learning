#!/usr/bin/env python3
"""Module for determining if a Markov chain is absorbing."""
import numpy as np


def absorbing(P):
    """
    Check if a Markov chain is absorbing.

    Args:
        P: Square 2D numpy.ndarray transition matrix of shape (n, n)

    Returns:
        True if chain is absorbing, False otherwise
    """
    conditions = [
        isinstance(P, np.ndarray) and len(P.shape) == 2,
        P.shape[0] == P.shape[1],
        np.allclose(np.sum(P, axis=1), 1)]

    if not all(conditions):
        return False

    diag = np.diag(P)
    absorbing_states = np.where(np.isclose(diag, 1))[0]
    if len(absorbing_states) == 0:
        return False

    n = P.shape[0]
    non_absorbing = list(set(range(n)) - set(absorbing_states))
    if len(non_absorbing) == 0:
        return True

    P_na = P[non_absorbing][:, non_absorbing]
    eye = np.eye(len(non_absorbing))
    try:
        N = np.linalg.inv(eye - P_na)
        return np.all(N >= 0)
    except np.linalg.LinAlgError:
        return False
