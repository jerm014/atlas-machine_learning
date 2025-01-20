#!/usr/bin/env python3
"""Calculate expectation step in Gaussian Mixture Model EM algorithm."""
import numpy as np


def expectation(X, pi, m, S):
    """Calculate expectation step for GMM.

    Args:
        X: numpy.ndarray shape (n, d) containing data set
        pi: numpy.ndarray shape (k,) containing cluster priors
        m: numpy.ndarray shape (k, d) containing centroid means
        S: numpy.ndarray shape (k, d, d) containing covariance matrices

    Returns:
        g: numpy.ndarray shape (k, n) with posterior probabilities
        l: total log likelihood
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(pi, np.ndarray) or len(pi.shape) != 1:
        return None, None
    if not isinstance(m, np.ndarray) or len(m.shape) != 2:
        return None, None
    if not isinstance(S, np.ndarray) or len(S.shape) != 3:
        return None, None
    if m.shape[0] != pi.shape[0] or S.shape[0] != pi.shape[0]:
        return None, None
    if m.shape[1] != X.shape[1] or S.shape[1] != X.shape[1]:
        return None, None
    if S.shape[1] != S.shape[2]:
        return None, None
    if not np.isclose(np.sum(pi), 1):
        return None, None

    k = pi.shape[0]
    if not np.isclose(np.sum(pi), 1):
        return None, None

    try:
        n = X.shape[0]
        pdfs = np.zeros((k, n))
        pdf = __import__('5-pdf').pdf

        for i in range(k):
            pdfs[i] = pdf(X, m[i], S[i])
            if pdfs[i] is None:
                return None, None

        numerator = pi[..., np.newaxis] * pdfs
        denominator = np.sum(numerator, axis=0)
        g = numerator / denominator
        L = np.sum(np.log(denominator))

        return g, L

    except Exception:
        return None, None
