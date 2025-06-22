#!/usr/bin/env python3
"""
This module provides a function to slice a Pandas DataFrame by extracting
specific columns and then selecting every Nth row from the extracted data.
"""


def slice(df):
    """
    Extracts the 'High', 'Low', 'Close', and 'Volume_BTC' columns from
    a pd.DataFrame and selects every 60th row from these columns.

    Args:
        df (pd.DataFrame): The input Pandas DataFrame.

    Returns:
        pd.DataFrame: The sliced DataFrame containing the specified
                      columns and rows.
    """
    # Define the list of columns to extract
    columns_to_extract = ['High', 'Low', 'Close', 'Volume_BTC']

    # Ensure all required columns exist in the DataFrame
    if not all(col in df.columns for col in columns_to_extract):
        missing = [col for col in columns_to_extract if col
                   not in df.columns]
        raise ValueError(f"Input DataFrame missing required columns: "
                         f"{missing}")

    # Extract the specified columns
    df_extracted = df[columns_to_extract]

    # Select every 60th row from the extracted DataFrame
    # Using .iloc for integer-location based indexing
    df_sliced = df_extracted.iloc[::60]

    return df_sliced
