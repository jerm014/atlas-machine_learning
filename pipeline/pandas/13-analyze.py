#!/usr/bin/env python3
"""
This module provides a function to compute descriptive statistics for
all numeric columns in a Pandas DataFrame, excluding the 'Timestamp'
column.
"""


def analyze(df):
    """
    Computes descriptive statistics for all columns of a pd.DataFrame,
    excluding the 'Timestamp' column.

    Args:
        df (pandas.DataFrame): The input Pandas DataFrame.

    Returns:
        pandas.DataFrame: A new DataFrame containing the descriptive
                          statistics.
    """
    df_copy = df.copy()

    # If 'Timestamp' column exists, drop it before computing statistics.
    # This ensures descriptive statistics are only calculated for
    # numerical data.
    if 'Timestamp' in df_copy.columns:
        df_for_analysis = df_copy.drop(columns=['Timestamp'])
    else:
        df_for_analysis = df_copy

    # Compute descriptive statistics.
    # .describe() by default computes for numeric columns.
    descriptive_statistics = df_for_analysis.describe()

    return descriptive_statistics
