+ACMAIQ-/usr/bin/env python3

+ACIAIgAi-
Wrie a function def pca(X, var+AD0--0.95): that performs PCA on a dataset:

 - X is a numpy.ndarray of shape (n, d) where:
 - n is the number of data points
 - d is the number of dimensions in each point

all dimensions have a mean of 0 across all data points

var is the fraction of the variance that the PCA transformation should
maintain

Returns: the weights matrix, W, that maintains var fraction of X+-IBg-s original
variance

W is a numpy.ndarray of shape (d, nd) where nd is the new dimensionality of
the transformed X
+ACIAIgAi-

import numpy as np


def pca(X, var+AD0--0.95):
    +ACIAIgAi-
    Performs Principal Component Analysis (PCA) on the input data.

    Args:
        X (numpy.ndarray):     The input data with shape (n, d).
        var (float, optional): The fraction of the variance that the PCA
                               transformation should maintain.
                               Defaults to 0.95.

    Returns:
        numpy.ndarray: The weights matrix, W, that maintains var fraction of
                       the original variance of X.
    +ACIAIgAi-
    +ACM- Step 1: Standardize the Data along the Features.
    X+AF8-std +AD0- (X - X.mean(axis +AD0- 0)) / x.std(axis +AD0- 0)

    +ACM- Step 2: Calculate the Covariance Matrix.
    cov +AD0- np.cov(X+AF8-std, ddof +AD0- 1, rowvar +AD0- False)

    +ACM- Step 3: Eigndecomposition on the Covariace Matrix.
    eigenvalues, eigenvectors +AD0- np.linalg.eig(cov)

    +ACM- Step 4: Sort the Principal Components.
    +ACM- (argsort returns lowest to highest. use ::-1 to reverse the list)
    order+AF8-of+AF8-importance +AD0- eigenvalues.argsort()+AFs-::-1+AF0-
    sorted+AF8-eigenvalues +AD0- eigenvalues+AFs-order+AF8-of+AF8-importance+AF0-
    +ACM- (sort the columns)
    sorted+AF8-eigenvectors +AD0- eigenvectors+AFs-:, order+AF8-of+AF8-importance+AF0-

    +ACM- Step 5: Compute the Explained Variance.
    explained+AF8-variance +AD0- sorted+AF8-eigenvalues / np.sum(sorted+AF8-eigenvalues)

    +ACM- Step 6: Reduce the Date via the Principal Components
    k +AD0- 2
    reduced+AF8-data +AD0- np.matmul(X+AF8-std, sorted+AF8-eigenvectors+AFs-:,k+AF0-)

    +ACM- Step 7: Determine the Explained Variance
    total+AF8-explained+AF8-variance +AD0- sum(explained+AF8-variance+AFs-:k+AF0-)

    

    +ACM- Compute the weights matrix, W, that maintains var fraction of the original
    +ACM- variance of X.
    W +AD0- sorted+AF8-eigenvectors+AFs-:, :k+AF0-

    return W
