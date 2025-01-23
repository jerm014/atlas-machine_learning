#!/usr/bin/env python3
"""Calculate probability density function of a Gaussian distribution."""
import numpy as np


def pdf(X, m, S):
    """Calculate PDF of a Gaussian distribution.

    Args:
        X: numpy.ndarray shape (n, d) with data points for PDF evaluation
        m: numpy.ndarray shape (d,) containing mean of distribution
        S: numpy.ndarray shape (d, d) containing covariance matrix

    Returns:
        numpy.ndarray shape (n,) with PDF values for each point
    """
    conditions = [
        isinstance(X, np.ndarray) and len(X.shape) == 2,
        isinstance(m, np.ndarray) and len(m.shape) == 1,
        isinstance(S, np.ndarray) and len(S.shape) == 2,
        X.shape[1] == m.shape[0] and X.shape[1] == S.shape[0],
        S.shape[0] == S.shape[1]
    ]

    if not all(conditions):
        return None

    try:
        d = X.shape[1]
        X_m = X - m
        inv_S = np.linalg.inv(S)
        det_S = np.linalg.det(S)

        exp_term = -0.5 * np.sum(
            np.matmul(X_m, inv_S) * X_m, axis=1)
        norm_const = 1 / (np.sqrt(((2 * np.pi) ** d) * det_S))
        P = np.maximum(norm_const * np.exp(exp_term), 1e-300)

        return P

    except Exception:
        return None
