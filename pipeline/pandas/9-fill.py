#!/usr/bin/env python3
"""
This module provides a function to clean and fill missing values in
a Pandas DataFrame, specifically for cryptocurrency historical data.
It handles column removal, forward filling, conditional filling,
and zero-filling for specific columns as descibed in task 9.
"""


def fill(df):
    """
    Removes the 'Weighted_Price' column, fills missing values in
    'Close' with the previous row’s value, fills 'High', 'Low', and
    'Open' with corresponding 'Close' values, and sets missing
    'Volume_(BTC)' and 'Volume_(Currency)' to 0.

    Args:
        df (pandas.DataFrame): The input Pandas DataFrame.

    Returns:
        pandas.DataFrame: The modified DataFrame.
    """
    # Create a copy to avoid modifying the original DataFrame directly
    # and prevent potential SettingWithCopyWarning.
    df_modified = df.copy()

    # 1. Remove the 'Weighted_Price' column.
    # Check if the column exists before attempting to drop it.
    if 'Weighted_Price' in df_modified.columns:
        df_modified = df_modified.drop(columns=['Weighted_Price'])

    # 2. Fill missing values in the 'Close' column with the previous
    #    row’s value (forward fill). This is crucial to do first,
    #    as other columns depend on 'Close' values.
    df_modified['Close'] = df_modified['Close'].fillna(method='ffill')

    # 3. Fill missing values in 'High', 'Low', and 'Open' columns with
    #    the corresponding 'Close' value in the same row.
    for col in ['High', 'Low', 'Open']:
        # Check if the column exists in the DataFrame.
        if col in df_modified.columns:
            df_modified[col] = df_modified[col].fillna(df_modified['Close'])

    # 4. Set missing values in 'Volume_(BTC)' and 'Volume_(Currency)' to 0.
    for col in ['Volume_(BTC)', 'Volume_(Currency)']:
        # Check if the column exists in the DataFrame.
        if col in df_modified.columns:
            df_modified[col] = df_modified[col].fillna(0)

    return df_modified
