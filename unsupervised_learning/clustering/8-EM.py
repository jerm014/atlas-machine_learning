#!/usr/bin/env python3
"""
Performs the expectation-maximization For a GMM with at most 1 loop.
"""
import numpy as np

initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """
    X is a numpy.ndarray of shape (n, d) containing the data set
    k is a positive integer containing the number of clusters
    iterations is a positive integer containing the max number of iterations
    tol is a non-negative float For tolerance of the log likelihood
    verbose is a bool that determines if info is printed about the algorithm

    Returns:
      pi, m, S, g, l
    or
      None, None, None, None, None on failure
    """

    conditions = [
        isinstance(X, np.ndarray) and len(X.shape) == 2,
        isinstance(k, int) and k > 0,
        isinstance(iterations, int) and iterations > 0,
        isinstance(tol, float) and tol >= 0,
        isinstance(verbose, bool)
    ]

    if not all(conditions):
        return None, None, None, None, None

    # Initialize parameters
    pi, m, S = initialize(X, k)
    if pi is None or m is None or S is None:
        return None, None, None, None, None

    # First E-step to get initial g and log-likelihood
    g, ll_old = expectation(X, pi, m, S)
    if g is None or ll_old is None:
        return None, None, None, None, None

    if verbose:
        log(0, ll_old)

    # Single loop For the EM iterations
    for i in range(1, iterations + 1):
        # M-step
        pi, m, S = maximization(X, g)
        if pi is None or m is None or S is None:
            return None, None, None, None, None

        # E-step
        g, ll_new = expectation(X, pi, m, S)
        if g is None or ll_new is None:
            return None, None, None, None, None

        # 1) Check convergence first
        if abs(ll_new - ll_old) <= tol:
            if verbose:
                log(i, ll_new)
            return pi, m, S, g, ll_new

        # 2) Otherwise, if not converged, print every 10 iterations
        if verbose and i % 10 == 0:
            log(i, ll_new)

        ll_old = ll_new

    # If we finish all iterations without breaking, print final result if
    # verbose
    if verbose and not i % 10 == 0:
        log(iterations, ll_old)

    return pi, m, S, g, ll_old


def log(i, v):
    """ print a log output with the iterations and a 5f value. """

    value_str = f"{v:.5f}".rstrip('0').rstrip('.')
    print(f"Log Likelihood after {i} iterations: {value_str}")
