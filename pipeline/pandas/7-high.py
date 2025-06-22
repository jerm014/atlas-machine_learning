#!/usr/bin/env python3
"""
This module provides a function to sort a Pandas DataFrame by the 'High'
price in descending order.
"""


def high(df):
    """
    Sorts the input Pandas DataFrame by the 'High' column in descending
    order.

    Args:
        df (pandas.DataFrame): The input Pandas DataFrame, expected to
                               contain a 'High' column.

    Returns:
        pandas.DataFrame: The DataFrame sorted by 'High' price in
                          descending order.
    """
    # Sort the DataFrame by the 'High' column in descending order
    # 'ascending=False' ensures descending order.
    # A KeyError will be raised if 'High' column does not exist,
    # which is appropriate behavior as per the problem description.
    sorted_df = df.sort_values(by='High', ascending=False)

    return sorted_df
