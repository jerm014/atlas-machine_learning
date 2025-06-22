#!/usr/bin/env python3
"""
This module provides a function to concatenate two Pandas DataFrames,
rearranging their MultiIndex to prioritize timestamp, filtering by a
specific timestamp range, and assigning hierarchical keys.
"""

import pandas as pd
index = __import__('10-index').index


def hierarchy(df1, df2):
    """
    Rearranges the MultiIndex so that Timestamp is the first level.
    Concatenates the bitstamp and coinbase tables from timestamps
    1417411980 to 1417417980, inclusvie
    Adds keys to the data, labeling rows from df2 as 'bitstamp' and
    rows from df1 as 'coinbase'.
    Ensures the data is displayed in chronological order.

    Args:
        df1 (pandas.DataFrame): The first input DataFrame (coinbase),
                                expected to have a 'Timestamp' column.
        df2 (pandas.DataFrame): The second input DataFrame (bitstamp),
                                expected to have a 'Timestamp' column.

    Returns:
        pandas.DataFrame: The concatenated and transformed DataFrame
                          with a hierarchical index.
    """
    # 1. Index both dataframes on their 'Timestamp' columns.
    indexed_df1 = index(df1)
    indexed_df2 = index(df2)

    # Define the timestamp range for filtering
    start_timestamp = 1417411980
    end_timestamp = 1417417980

    # 2. Filter both dataframes to include timestamps within the specified
    #    range (inclusive).
    filtered_df1 = indexed_df1[
        (indexed_df1.index >= start_timestamp) &
        (indexed_df1.index <= end_timestamp)
    ]
    filtered_df2 = indexed_df2[
        (indexed_df2.index >= start_timestamp) &
        (indexed_df2.index <= end_timestamp)
    ]

    # 3. Concatenate the selected rows.
    # 4. Add keys to the data, labeling rows from df2 as 'bitstamp' and
    #    rows from df1 as 'coinbase'.
    concatenated_df = pd.concat(
                                [filtered_df2,
                                filtered_df1],
                                keys=['bitstamp', 'coinbase'])

    # 5. Rearrange the MultiIndex so that Timestamp is the first level.
    concatenated_df = concatenated_df.swaplevel(0, 1)

    # 6. Ensure the data is displayed in chronological order.
    concatenated_df = concatenated_df.sort_index()

    return concatenated_df
