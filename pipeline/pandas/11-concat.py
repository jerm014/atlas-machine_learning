#!/usr/bin/env python3
"""
This module provides a function to concatenate two Pandas DataFrames
after indexing them, filtering one by timestamp, and assigning keys
to the concatenated rows.
"""

import pandas
index = __import__('10-index').index


def concat(df1, df2):
    """
    Indexes both dataframes on their 'Timestamp' columns.
    Includes all timestamps from df2 (bitstamp) up to and including
    timestamp 1417411920.
    Concatenates the selected rows from df2 to the top of df1.
    Adds keys to the concatenated data, labeling the rows from df2 as
    'bitstamp' and the rows from df1 as 'coinbase'.

    Args:
        df1 (pandas.DataFrame): The first input DataFrame (coinbase),
                                expected to have a 'Timestamp' column.
        df2 (pandas.DataFrame): The second input DataFrame (bitstamp),
                                expected to have a 'Timestamp' column.

    Returns:
        pandas.DataFrame: The concatenated DataFrame.
    """
    # 1. Index both dataframes on their 'Timestamp' columns.
    indexed_df1 = index(df1)
    indexed_df2 = index(df2)

    # 2. Include all timestamps from df2 (bitstamp) up to and including
    #    timestamp 1417411920.
    target_timestamp = 1417411920
    filtered_df2 = indexed_df2[indexed_df2.index <= target_timestamp]

    # 3. Concatenate the selected rows from df2 to the top of df1.
    # 4. Adds keys to the concatenated data, labeling the rows from df2
    #    as 'bitstamp' and the rows from df1 as 'coinbase'.
    concatenated_df = pandas.concat(
        [filtered_df2, indexed_df1],
        keys=['bitstamp', 'coinbase']
    )

    return concatenated_df
