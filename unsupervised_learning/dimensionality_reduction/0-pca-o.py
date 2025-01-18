+ACMAIQ-/usr/bin/env python3

+ACIAIgAi-
Wrie a function def pca(X, var+AD0-0.95): that performs PCA on a dataset:

 - X is a numpy.ndarray of shape (n, d) where:
 - n is the number of data points
 - d is the number of dimensions in each point

all dimensions have a mean of 0 across all data points

var is the fraction of the variance that the PCA transformation should
maintain

Returns: the weights matrix, W, that maintains var fraction of X+IBg-s original
variance

W is a numpy.ndarray of shape (d, nd) where nd is the new dimensionality of
the transformed X
+ACIAIgAi-

import numpy as np


def pca(X, var+AD0-0.95):
    +ACIAIgAi-
    Performs Principal Component Analysis (PCA) on the input data.

    Args:
        X (numpy.ndarray): The input data with shape (n, d).
        var (float, optional): The fraction of the variance that the PCA
            transformation should maintain. Defaults to 0.95.

    Returns:
        numpy.ndarray: The weights matrix, W, that maintains var fraction of
            the original variance of X.
    +ACIAIgAi-
    +ACM- Center the data by subtracting the mean
    X+AF8-mean +AD0- X - np.mean(X, axis+AD0-0)

    +ACM- Compute the covariance matrix
    cov +AD0- np.cov(X+AF8-mean.T)

    +ACM- Compute the eigenvalues and eigenvectors of the covariance matrix
    eigenvalues, eigenvectors +AD0- np.linalg.eig(cov)

    +ACM- Sort the eigenvalues and eigenvectors in descending order
    idx +AD0- eigenvalues.argsort()+AFs-::-1+AF0-
    eigenvalues +AD0- eigenvalues+AFs-idx+AF0-
    eigenvectors +AD0- eigenvectors+AFs-:, idx+AF0-

    +ACM- Compute the cumulative explained variance ratio
    cumulative+AF8-variance+AF8-ratio +AD0- np.cumsum(eigenvalues) / np.sum(eigenvalues)

    +ACM- Determine the number of dimensions to keep based on the desired variance
    n+AF8-components +AD0- np.argmax(cumulative+AF8-variance+AF8-ratio +AD4APQ- var) +- 1

    +ACM- Compute the weights matrix, W
    W +AD0- eigenvectors+AFs-:, :n+AF8-components+AF0-

    return W
