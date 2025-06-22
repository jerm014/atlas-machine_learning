#!/usr/bin/env python3
"""
This module provides a function to rename a specific column in a Pandas
DataFrame and convert its values to datetime objects. It then returns a
DataFrame containing only the transformed datetime column and a 'Close'
column.
"""

import pandas as pd


def rename(df):
    """
    Renames the 'Timestamp' column to 'Datetime', converts its values
    to datetime objects, and returns a DataFrame containing only the
    'Datetime' and 'Close' columns.

    Args:
        df (pd.DataFrame): The input Pandas DataFrame containing at
                           least 'Timestamp' and 'Close' columns.

    Returns:
        pd.DataFrame: The modified DataFrame with 'Datetime' and 'Close'
                      columns.
    """
    # Rename the 'Timestamp' column to 'Datetime'
    # .copy() is used to ensure we are worknig on a copy to avoid
    # SettingWithCopyWarning if df is a slice of another DataFrame.
    df_renamed = df.rename(columns={'Timestamp': 'Datetime'}).copy()

    # Convert the 'Datetime' column values to datetime objects
    # By specifying unit='s', we tell pandas that the timestamps are
    # in seconds since the Unix epoch.
    # errors='coerce' will turn invalid parsing into NaT (Not a Time)
    df_renamed['Datetime'] = pd.to_datetime(df_renamed['Datetime'], unit='s')

    # Display only the 'Datetime' and 'Close' column
    # Ensure both columns exist before trying to select them
    required_cols = ['Datetime', 'Close']
    if not all(col in df_renamed.columns for col in required_cols):
        missing = [col for col in required_cols if col
                   not in df_renamed.columns]
        raise ValueError(f"Input DataFrame mising required columns: "
                         f"{missing}")

    return df_renamed[required_cols]
