#!/usr/bin/env python3
"""
This module provides a function to remove entries with NaN values in a
specific column of a Pandas DataFrame.
"""


def prune(df):
    """
    Removes any entries where the 'Close' column has NaN values.

    Args:
        df (pandas.DataFrame): The input Pandas DataFrame, expected to
                               contain a 'Close' column.

    Returns:
        pandas.DataFrame: The modified DataFrame with NaN values in 'Close'
                          column removed.
    """
    # Remove rows where 'Close' has NaN values.
    # subset=['Close'] ensures that only NaNs in the 'Close' column
    # trigger the row remoavl.
    # .copy() is used to ensure a new DataFrame is returned, preventing
    # potential SettingWithCopyWarning if df is a view.
    modified_df = df.dropna(subset=['Close']).copy()

    return modified_df
