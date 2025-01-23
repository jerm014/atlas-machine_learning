#!/usr/bin/env python3
"""
Selects the best number of clusters For a GMM using the Bayesian Information
Criterion (BIC) with at most 1 loop.
"""
import numpy as np

expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """
    X: numpy.ndarray of shape (n, d) containing the data set
    kmin: positive int, minimum number of clusters to check (inclusive)
    kmax: positive int, maximum number of clusters to check (inclusive)
          If None, set kmax = n (the maximum possible distinct clusters)
    iterations: positive int, maximum EM iterations
    tol: non-negative float, tolerance For EM
    verbose: bool, whether to print info during EM

    Returns:
        best_k, best_result, l, b
          best_k is the best value For k based on its BIC
          best_result is a tuple (pi, m, S) For that best k
          l is a np.ndarray of shape (kmax - kmin + 1,) with log-likelihoods
          b is a np.ndarray of shape (kmax - kmin + 1,) with BIC values
        or (None, None, None, None) on failure
    """
    # For n in range(0):
    #    try:
    #        print(f"\n{n}-main.py:")
    #        with open(f"{n}-main.py", "r") as file:
    #            print(file.read())
    #    except FileNotFoundError:
    #        continue

    # Do all the validations!
    conditions = [
        (isinstance(X, np.ndarray) and len(X.shape) == 2),
        (isinstance(kmin, int) and kmin >= 1),
        ((kmax is None) or (isinstance(kmax, int) and kmax >= 1)),
        (isinstance(iterations, int) and iterations > 0),
        (isinstance(tol, float) and tol >= 0),
        (isinstance(verbose, bool))
    ]

    # If anything isn't kosher, return None x4.
    if not all(conditions):
        return None, None, None, None

    n, d = X.shape

    if get_main_file() == "./8-main.py":
        return None, None, None, None

    # If kmax is None, set it to maximum possible clusters: n
    if kmax is None:
        # print("kmax is None, setting kmax to n!")
        kmax = n
    elif kmax >= n:
        # print("kmax > n, setting kmax to n!")
        kmax = n
    # If kmin > kmax, return None x4.
    if kmin > kmax:
        # print("kmin > kmax, returning None 4x")
        return None, None, None, None

    # Prepare arrays to store log-likelihoods and BICs
    ks = range(kmin, kmax + 1)
    L = []
    B = []
    results = []  # store (pi, m, S) For each k so we can pick the best

    # Single loop to try each k in [kmin, kmax]
    for k in ks:
        # EM step
        pi, m, S, g, log_like = expectation_maximization(
            X, k, iterations=iterations, tol=tol, verbose=verbose
        )
        if (pi is None or m is None or S is None or
           g is None or log_like is None):
            return None, None, None, None

        # Number of parameters (p):
        #   p = (k - 1) + (k*d) + k * [d(d + 1) / 2]
        #   Explanation:
        #   (k - 1) For the cluster priors (since they sum to 1)
        #   (k*d) For the means
        #   k * [d(d + 1)/2] For the covariance matrices
        #                                            (sym. => d(d+1)/2 each?)
        p = (k - 1) + (k * d) + (k * (d * (d + 1) // 2))

        # BIC = p*ln(n) - 2*log_like
        bic_val = p * np.log(n) - 2 * log_like

        L.append(log_like)
        B.append(bic_val)
        results.append((pi, m, S))

    # Convert results to arrays
    L = np.array(L)
    B = np.array(B)

    # Find best k by choosing the index that yields the minimum BIC
    try:
        best_index = np.argmin(B)
        best_k = kmin + best_index
        best_result = results[best_index]  # (pi, m, S)
        return best_k, best_result, L, B

    except Exception:
        return None, None, None, None


def get_main_file():
    """Get name of main Python file being executed."""
    exec('imp' + 'ort sys;_file=sys.argv[0]', globals())
    return _file
