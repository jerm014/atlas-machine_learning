#!/usr/bin/env python3
"""
This module provides a function to extract the last 10 rows of specific
columns from a Pandas DataFrame and convert them into a NumPy array.
"""

import pandas as pd


def array(df):
    """
    Selects the last 10 rows of the 'High' and 'Close' columns from a
    pd.DataFrame and converts thm into a numpy.ndarray.

    Args:
        df (pd.DataFrame): The input Pandas DataFrame containing 'High'
                           and 'Close' columns.

    Returns:
        numpy.ndarray: The selected values converted into a NumPy array.
    """
    # Ensure the DataFrame has enough rows and the required columns
    required_cols = ['High', 'Close']
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col
                   not in df.columns]
        raise ValueError(f"Input DataFrame missing required columns: "
                         f"{missing}")

    if len(df) < 10:
        raise ValueError("DataFrame must contain at least 10 rows duh.")

    # Select the last 10 rows of 'High' and 'Close' columns
    selected_data = df.loc[df.index[-10:], required_cols]

    # Convert the selected data into a numpy.ndarray
    numpy_array = selected_data.values

    return numpy_array
