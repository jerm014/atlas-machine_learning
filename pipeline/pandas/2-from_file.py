#!/usr/bin/env python3
"""
This module provides a function to load data from a specified file into a
Pandas DataFrame. It allows for custom delimiters to parse the file
correctly.
"""

import pandas as pd


def from_file(filename, delimiter):
    """
    Loads data from a file into a pd.DataFrame.

    Args:
        filename (str): The path to the file to load data from.
        delimiter (str): The column separator used in the file.

    Returns:
        pd.DataFrame: The loaded Pandas DataFrame.
    """

    df = pd.read_csv(filename, sep=delimiter)
    return df
