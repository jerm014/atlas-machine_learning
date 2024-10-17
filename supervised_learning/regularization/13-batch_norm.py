#!/usr/bin/env python3
""" module for bacth_norm function for task 13 of project 2297 """
import numpy as np

def batch_norm(Z, gamma, beta, epsilon):
  """
  documentation goes here.
  """
    # Calculate mean of each feature
    mu = np.mean(Z, axis=0)
    
    # Calculate variance of each feature
    var = np.var(Z, axis=0)
    
    # Normalize Z 
    Z_norm = (Z - mu) / np.sqrt(var + epsilon)
    
    # Scale and shift the normalized values
    Z_tilde = gamma * Z_norm + beta
    
    return Z_tilde
