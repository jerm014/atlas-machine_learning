#!/usr/bin/env python3
"""
Yo dawg, this module's all about that sweet Bayesian optimization life.
Get ready to optimize ur black-box functions like a b0ss!
"""
import numpy as np
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
