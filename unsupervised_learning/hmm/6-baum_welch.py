#!/usr/bin/env python3
"""Module for various Hidden Markov Model algorithms (forward, backward,
etc.)."""
import numpy as np


def baum_welch(Observations, Transition, Emission, Initial, iterations=1000):
    """
    Perform Baum-Welch algorithm for Hidden Markov Model parameter
    estimation.

    Args:
        Observations: numpy.ndarray of shape (T,) containing observation
                      indices
        Transition:   numpy.ndarray of shape (M, M) with transition
                      probabilities
        Emission:     numpy.ndarray of shape (M, N) with emission
                      probabilities
        Initial:      numpy.ndarray of shape (M, 1) with starting
                      probabilities
        iterations:   Number of expectation-maximization iterations

    Returns:  Transition, Emission
        Transition: Converged transition probabilities matrix
        Emission:   Converged emission probabilities matrix
        or None, None on failure
    """
    conditions = [
        isinstance(Observations, np.ndarray),
        isinstance(Transition, np.ndarray),
        isinstance(Emission, np.ndarray),
        isinstance(Initial, np.ndarray)
    ]

    if not all(conditions):
        return None, None

    conditions = [
        len(Transition.shape) == 2,
        len(Initial.shape) == 2,
        len(Emission.shape) == 2,
        Transition.shape[0] == Transition.shape[1],
        Initial.shape[1] == 1,
        Initial.shape[0] == Transition.shape[0],
        np.allclose(np.sum(Transition, axis=1), 1),
        np.allclose(np.sum(Emission, axis=1), 1),
        np.allclose(np.sum(Initial), 1)
    ]

    if not all(conditions):
        return None, None

    try:

        M = Transition.shape[0]
        T = Observations.shape[0]
        N = Emission.shape[1]

        for _ in range(iterations):
            alpha = np.zeros((M, T))
            beta = np.zeros((M, T))

            # Forward pass
            alpha[:, 0] = Initial.flatten() * Emission[:, Observations[0]]
            for t in range(1, T):
                for j in range(M):
                    alpha[j, t] = Emission[j, Observations[t]] * np.sum(
                        alpha[:, t-1] * Transition[:, j]
                    )

            # Backward pass
            beta[:, -1] = 1
            for t in range(T-2, -1, -1):
                for j in range(M):
                    beta[j, t] = np.sum(
                        Transition[j, :] * Emission[:, Observations[t+1]] *
                        beta[:, t+1]
                    )

            xi = np.zeros((M, M, T-1))
            for t in range(T-1):
                denominator = np.sum(
                    alpha[:, t].reshape((-1, 1)) * Transition *
                    Emission[:, Observations[t+1]].reshape((1, -1)) *
                    beta[:, t+1].reshape((1, -1))
                )
                for i in range(M):
                    numerator = alpha[i, t] * Transition[i, :] * \
                        Emission[:, Observations[t+1]] * beta[:, t+1]
                    xi[i, :, t] = numerator / denominator

            gamma = np.sum(xi, axis=1)
            Transition = np.sum(xi, 2) / np.sum(gamma, axis=1).reshape((-1, 1))

            gamma = np.hstack((gamma, np.sum(
                xi[:, :, T-2], axis=0).reshape((-1, 1))
            ))

            denominator = np.sum(gamma, axis=1)
            for ll in range(N):
                Emission[:, ll] = np.sum(gamma[:, Observations == ll], axis=1)

            Emission = np.divide(
                Emission, denominator.reshape((-1, 1)),
                out=np.zeros_like(Emission),
                where=denominator.reshape((-1, 1)) != 0
            )

        return Transition, Emission

    except Exception:
        return None, None
