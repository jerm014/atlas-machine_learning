#!/usr/bin/env python3
"""
Yo dawg, this module's all about that sweet Bayesian optimization life.
Get ready to optimize ur black-box functions like a b0ss!
"""
import numpy as np
from scipy.stats import norm
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """
    1337 Bayesian optimization for that noiseless 1D Gaussian process swag.

    This class is straight fire for finding optimal points in ur function
    space. Get rdy to minimize/maximize like a pr0! Woot!
    """

    def __init__(self, f, X_init, Y_init, bounds, ac_samples, l=1,
                 sigma_f=1, xsi=0.01, minimize=True):
        """
        Initialize the sickest Bayesian optimization setup u've ever seen!

        Args:
            f:          Ur black-box function 2 be optimized (2 l33t 2 peek
                        inside)
            X_init:     Initial X inputs matrix of shape (t, 1)
            Y_init:     Initial Y outputs matrix of shape (t, 1)
            bounds:     Tuple of (min, max) 4 searching optimal point
            ac_samples: Number of samples 4 acquisition analysis
            l:          Length param 4 the kernel (default=1)
            sigma_f:    Standard deviation 4 black-box function (default=1)
            xsi:        Exploration-exploitation factor (default=0.01)
            minimize:   True 4 minimization, False 4 maximization
                        (default=True)
        """
        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)

        # Generate acquisition sample points
        min_bound, max_bound = bounds
        self.X_s = np.linspace(min_bound,
                               max_bound,
                               ac_samples).reshape(-1, 1)

        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        """
        Calcs fo next best sample location using Expected Improvement (EI).

        Returns:
            tuple: X_next (1,) array of next best point,
                  EI (ac_samples,) array of expected improvements
        """
        mu, sigma = self.gp.predict(self.X_s)

        # Get current best depending on minimization/maximization
        if self.minimize:
            mu_sample_opt = np.min(self.gp.Y)
        else:
            mu_sample_opt = np.max(self.gp.Y)

        # Needed for stability
        sigma = np.maximum(sigma, 1e-9)

        # Calculate improvement based on optimization type
        with np.errstate(divide='warn'):
            if self.minimize:
                imp = mu_sample_opt - mu - self.xsi
            else:
                imp = mu - mu_sample_opt - self.xsi

            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)

            # Set EI to 0 where sigma is 0
            ei[sigma == 0] = 0

        # Find index of best EI
        X_next = self.X_s[np.argmax(ei)]

        return X_next, ei

    def optimize(self, iterations=100):
        """
        Optimize that black-box function like a boss!

        Args:
            iterations: Max number of iterations 2 perform (default=100)

        Returns:
            tuple: X_opt (1,) optimal point array,
                  Y_opt (1,) optimal value array
        """
        for _ in range(iterations):
            X_next, _ = self.acquisition()

            # Check if point was already sampled
            if np.any(np.abs(X_next - self.gp.X) <= 1e-10):
                break

            # Sample new point and update GP
            Y_next = self.f(X_next)
            self.gp.update(X_next.reshape(-1, 1), Y_next.reshape(-1, 1))

        # Get optimal point based on minimization/maximization
        if self.minimize:
            idx_opt = np.argmin(self.gp.Y)
        else:
            idx_opt = np.argmax(self.gp.Y)

        X_opt = self.gp.X[idx_opt]
        Y_opt = self.gp.Y[idx_opt]

        return X_opt, Y_opt
