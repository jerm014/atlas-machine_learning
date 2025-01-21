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

    You may use at most 1 loop. Returns:
      pi, m, S, g, l
    or
      None, None, None, None, None on failure
    """
    # Basic validations
    if (not isinstance(X, np.ndarray) or len(X.shape) != 2 or
       not isinstance(k, int) or k <= 0 or
       not isinstance(iterations, int) or iterations <= 0 or
       not isinstance(tol, float) or tol < 0 or
       not isinstance(verbose, bool)):
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
        print(f"Log Likelihood after 0 iterations: {ll_old:.5f}")

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

        # Check convergence
        if abs(ll_new - ll_old) <= tol:
            if verbose:
                print(f"Log Likelihood after {i} iterations: {ll_new:.5f}")
            return pi, m, S, g, ll_new

        # Verbose output every 10 iterations
        if verbose and i % 10 == 0:
            print(f"Log Likelihood after {i} iterations: {ll_new:.5f}")

        ll_old = ll_new

    # If we finish all iterations without breaking, print final result if
    # verbose
    if verbose:
        print(f"Log Likelihood after {iterations} iterations: {ll_old:.5f}")

    return pi, m, S, g, ll_old
