#!/usr/bin/env python3
"""specificity function for project 2295 task 3"""

import numpy as np


def specificity(confusion):
  """
  calculate the specificity for each class in a confusion matrix
  
  Args:
    confusion: numpy.ndarray of shape (classes, classes)
      row indices represent the correct labels
      column indices represent the predicted labels
      classes is the number of classes
  Returns:
    numpy.ndarray of shape (classes,) containing the specificity of each class

  Notes from PLD:
    Specificity is true negative rate tn / tn + fp (minimize fp)
  """
  return np.sum(~y_true & ~y_pred) / np.sum(~y_true)
