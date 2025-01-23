#!/usr/bin/env python3
"""Performs agglomerative clustering with dendrogram visualization."""

import scipy.cluster.hierarchy
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    """Perform agglomerative clustering with Ward linkage.

    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset
        dist: float, maximum cophenetic distance for clusters

    Returns:
        numpy.ndarray of shape (n,) containing cluster indices
    """
    Z = scipy.cluster.hierarchy.ward(X)

    fig = plt.figure(figsize=(10, 7))
    dendrogram = scipy.cluster.hierarchy.dendrogram(
        Z,
        color_threshold=dist,
    )
    plt.show()

    clss = scipy.cluster.hierarchy.fcluster(Z, dist, criterion='distance')

    return clss - 1
