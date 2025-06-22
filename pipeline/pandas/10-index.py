#!/usr/bin/env python3
"""
This module provides a function to set a specified column of a Pandas
DataFrame as its index.
"""


def index(df):
    """
    Sets the 'Timestamp' column as the index of the DataFrame.

    Args:
        df (pandas.DataFrame): The input Pandas DataFrame, expected to
                               contain a 'Timestamp' column.

    Returns:
        pandas.DataFrame: The modified DataFrame with 'Timestamp'
                          set as its index.
    """
    # Check if 'Timestamp' column exists before attempting to set it
    # as index.
    if 'Timestamp' not in df.columns:
        raise ValueError("Input DataFrame must contain a 'Timestamp' column.")

    # Set the 'Timestamp' column as the index.
    # inplace=False by default, so it returns a new DataFrame.
    # drop=True by default, so it removes the column from the DataFrame.
    modified_df = df.set_index('Timestamp')

    return modified_df
