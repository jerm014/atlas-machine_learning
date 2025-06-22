#!/usr/bin/env python3
"""
This module provides a function to convert a NumPy array into a Pandas DataFrame.
The conversion ensures that the DataFrame columns are labeled alphabetically
and capitalized, adhering to a limit of 26 columns (A-Z).
"""

import pandas as pd


def from_numpy(array):
    """
    Creates a pd.DataFrame from a np.ndarray.

    The columns of the pd.DataFrame are labeled in alphabetical order and
    capitalized. There will not be more than 26 columns.

    Args:
        array: The np.ndarray from which to create the pd.DataFrame.

    Returns:
        The newly created pd.DataFrame.
    """

    # Get the number of columns from the input numpy array
    # array.shape[1] implicitly assumes a 2D array, which is typical
    # for DataFrame creation from an array.
    num_columns = array.shape[1]

    # Check if the number of columns exceeds the limit (A-Z)
    if num_columns > 26:
        # This case is excluded by the problem description, but a robust
        # function might handle it (e.g., raise an error or use AA, AB,
        # etc.). For this problem, we can assume it won't happen.
        # However, for completeness, we'll add a check.
        raise ValueError("Number of columns exceeds 26 (A-Z) limit for "
                         "labeling.")

    # Generate column labels in alphabetical order (A, B, C, ...)
    # ASCII value for 'A' is 65.
    column_labels = [chr(65 + i) for i in range(num_columns)]

    # Create the pandas DataFrame using the numpy array and generated
    # column labels
    df = pd.DataFrame(array, columns=column_labels)

    return df
