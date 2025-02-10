import numpy as np

class GaussianProcess:
    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """
        Initialize the Gaussian Process model.
        
        Args:
            X_init (numpy.ndarray): Input samples of shape (t, 1)
            Y_init (numpy.ndarray): Output samples of shape (t, 1)
            l (float): Length parameter for the RBF kernel
            sigma_f (float): Standard deviation for outputs
        """
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        
        # Calculate initial covariance matrix
        self.K = self.kernel(X_init, X_init)
        
    def kernel(self, X1, X2):
        """
        Calculate the RBF (squared exponential) kernel between two sets of points.
        
        Args:
            X1 (numpy.ndarray): First set of points of shape (m, 1)
            X2 (numpy.ndarray): Second set of points of shape (n, 1)
            
        Returns:
            numpy.ndarray: Covariance kernel matrix of shape (m, n)
        """
        # Compute pairwise squared distances
        sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
        
        # Return the RBF kernel
        return self.sigma_f**2 * np.exp(-0.5 / self.l**2 * sqdist)

    def predict(self, X_s):
        """
        Predict the mean and standard deviation of points in a Gaussian process.
        
        Args:
            X_s (numpy.ndarray): Points to predict of shape (s, 1)
            
        Returns:
            tuple:
                mu (numpy.ndarray): Mean for each point of shape (s,)
                sigma (numpy.ndarray): Variance for each point of shape (s,)
        """
        # Calculate kernel between X_s and X_train
        K_s = self.kernel(self.X, X_s)
        
        # Calculate kernel for X_s
        K_ss = self.kernel(X_s, X_s)
        
        # Calculate mean (mu)
        K_inv = np.linalg.inv(self.K)
        mu = K_s.T.dot(K_inv).dot(self.Y).reshape(-1)
        
        # Calculate variance (sigma)
        sigma = np.diag(K_ss - K_s.T.dot(K_inv).dot(K_s))
        
        return mu, sigma