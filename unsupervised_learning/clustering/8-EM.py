#!/usr/bin/env python3
"""Perform expectation maximization 4 a Gaussian Mixture Model."""
import numpy as np


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """Run EM algorithm for GMM.

    Args:
        X: numpy.ndarray shape (n, d) containing data set
        k: positive integer for number of clusters
        iterations: maximum number of iterations for the algorithm
        tol: non-negative tolerance for early stopping
        verbose: boolean to determine info printing

    Returns:
        pi: array shape (k,) with final cluster priors
        m: array shape (k, d) with final means
        S: array shape (k, d, d) with final covariances
        g: array shape (k, n) with final probabilities
        l: final log likelihood
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None, None
    if not isinstance(k, int) or k <= 0 or k >= X.shape[0]:
        return None, None, None, None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None, None
    if not isinstance(tol, float) or tol < 0:
        return None, None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None, None

    initialize = __import__('4-initialize').initialize
    expectation = __import__('6-expectation').expectation
    maximization = __import__('7-maximization').maximization

    try:
        pi, m, S = initialize(X, k)
        if pi is None:
            return None, None, None, None, None

        l_prev = 0
        for i in range(iterations):
            g, L = expectation(X, pi, m, S)
            if g is None:
                return None, None, None, None, None

            if verbose and (i % 10 == 0 or i == iterations - 1):
                print('Log Likelihood after {} iterations: {}'.format(
                    i, round(L, 5)))

            if abs(L - l_prev) <= tol:
                break

            pi, m, S = maximization(X, g)
            if pi is None:
                return None, None, None, None, None

            l_prev = L

        g, L = expectation(X, pi, m, S)
        if g is None:
            return None, None, None, None, None

        if verbose:
            print('Log Likelihood after {} iterations: {}'.format(
                i + 1, round(L, 5)))

        return pi, m, S, g, L

    except Exception:
        return None, None, None, None, None
