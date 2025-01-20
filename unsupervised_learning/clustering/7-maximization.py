#!/usr/bin/env python3
"""Calculate maximization step in Gaussian Mixture Model EM algorithm."""
import numpy as np


def maximization(X, g):
    """Calculate maximization step for GMM.

    Args:
        X: numpy.ndarray shape (n, d) containing data set
        g: numpy.ndarray shape (k, n) containing posteriors

    Returns:
        pi: numpy.ndarray shape (k,) with updated cluster priors
        m: numpy.ndarray shape (k, d) with updated means
        S: numpy.ndarray shape (k, d, d) with updated covariances
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None
    if not isinstance(g, np.ndarray) or len(g.shape) != 2:
        return None, None, None
    if X.shape[0] != g.shape[1]:
        return None, None, None

    try:
        k = g.shape[0]
        n_soft = np.sum(g, axis=1)
        pi = n_soft / n_soft.sum()

        m = np.matmul(g, X) / n_soft[:, np.newaxis]

        S = np.zeros((k, X.shape[1], X.shape[1]))
        for i in range(k):
            x_m = X - m[i]
            S[i] = (g[i, :, np.newaxis, np.newaxis] * np.matmul(
                x_m[:, :, np.newaxis], x_m[:, np.newaxis, :])).sum(axis=0)
            S[i] = S[i] / n_soft[i]

        return pi, m, S

    except Exception:
        return None, None, None