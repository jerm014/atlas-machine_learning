#!/usr/bin/env python3
"""
Bruh, this script's gonna optimize ur ML model like it's no one's business!
Get ready for some sick hyperparameter tuning with GPyOpt!
"""
import numpy as np
import GPyOpt
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import pickle


def create_model(params):
    """
    Creates a GMM model with them sweet hyperparams.

    Args:
        params: Dict of hyperparams 2 try
    Returns:
        float: That dope silhouette score
    """
    # Extract params from the optimization suggestion
    n_components = int(params[0][0])
    covariance_type_idx = int(params[0][1])
    tol = float(params[0][2])
    reg_covar = float(params[0][3])
    max_iter = int(params[0][4])

    # Map covariance type index to actual type
    cov_types = ['full', 'tied', 'diag', 'spherical']
    covariance_type = cov_types[covariance_type_idx]

    # Generate some sick synthetic data
    X, _ = make_blobs(n_samples=1000, centers=3, n_features=2)

    # Initialize that GMM
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        tol=tol,
        reg_covar=reg_covar,
        max_iter=max_iter,
        random_state=42
    )

    # Fit and score the model
    gmm.fit(X)
    if not gmm.converged_:
        return -1

    labels = gmm.predict(X)
    score = silhouette_score(X, labels)

    # Save checkpoint if it's the best so far
    filename = f"gmm_ncompoents-{n_components}_" \
               f"covariancetype-{covariance_type}_tol-{tol:.2e}_" \
               f"regcovar-{reg_covar:.2e}" \
               f"_maxiter-{max_iter}.pkl"
    with open(filename, 'wb') as f:
        pickle.dump(gmm, f)

    return score


def optimize_gmm():
    """
    Optimizes them GMM hyperparams using Bayesian optimization.
    """
    # Define the parameter space
    param_space = [
        {'name': 'n_components', 'type': 'discrete',
         'domain': range(2, 10)},
        {'name': 'covariance_type', 'type': 'discrete',
         'domain': range(4)},
        {'name': 'tol', 'type': 'continuous',
         'domain': (1e-5, 1e-2)},
        {'name': 'reg_covar', 'type': 'continuous',
         'domain': (1e-8, 1e-4)},
        {'name': 'max_iter', 'type': 'discrete',
         'domain': range(50, 301, 10)}
    ]

    # Initialize optimizer
    optimizer = GPyOpt.methods.BayesianOptimization(
        f=create_model,
        domain=param_space,
        acquisition_type='EI',
        maximize=True
    )

    # Run optimization
    optimizer.run_optimization(max_iter=30)

    # Plot convergence
    optimizer.plot_convergence("convergence.png")

    # Save report
    with open('bayes_opt.txt', 'w') as f:
        f.write("Bayesian Optimization Report\n")
        f.write("==========================\n\n")
        f.write(f"Best score: {optimizer.fx_opt}\n")
        f.write("Best parameters:\n")
        for param, value in zip(param_space, optimizer.x_opt):
            f.write(f"{param['name']}: {value}\n")


if __name__ == "__main__":
    optimize_gmm()
