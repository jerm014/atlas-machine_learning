#!/usr/bin/env python3
"""
This module provides a function to sort a Pandas DataFrame in reverse
chronological order and then transpose it.
"""


def flip_switch(df):
    """
    Sorts the data in reverse chronological order and transposes the
    sorted DataFrame.

    Assumes a 'Datetime' column exists for chronological sorting.

    Args:
        df (pandas.DataFrame): The input Pandas DataFrame.

    Returns:
        pandas.DataFrame: The transformed DataFrame, sorted and transposed.
    """
    sorted_df = df.sort_values(by='Datetime', ascending=False)
    transformed_df = sorted_df.T

    return transformed_df
