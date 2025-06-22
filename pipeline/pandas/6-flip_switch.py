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
    try:
        # Sort the DataFrame by the 'Datetime' column in descending order
        sorted_df = df.sort_values(by='Datetime', ascending=False)
    except KeyError:
        # If 'Datetime' column does not exist, try sorting by index?
        sorted_df = df.sort_index(ascending=False)

    # Transpose the sorted DataFrame
    transformed_df = sorted_df.T

    return transformed_df
