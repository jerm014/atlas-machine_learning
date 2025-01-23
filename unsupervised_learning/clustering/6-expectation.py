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
    conditions = [[
        isinstance(X, np.ndarray) and len(X.shape) == 2,
        isinstance(pi, np.ndarray) and len(pi.shape) == 1,
        isinstance(m, np.ndarray) and len(m.shape) == 2,
        isinstance(S, np.ndarray) and len(S.shape) == 3
    ],[
        m.shape[0] == pi.shape[0],
        S.shape[0] == pi.shape[0],
        m.shape[1] == X.shape[1],
        S.shape[1] == X.shape[1],
        S.shape[1] == S.shape[2],
        np.isclose(np.sum(pi), 1)
    ]]

    # don't check the second half of conditions unless the first half passes
    if not all(conditions[0]) and not all(conditions[1]):
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
