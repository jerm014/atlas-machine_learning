#!/usr/bin/env python3
"""
This script demonstrates how to create a Pandas DataFrame from a dictionary,
specifying column names, values, and row (index) labels.
"""

import pandas as pd

# Define the data using a dictionary for column values
data = {
    'First': [0.0, 0.5, 1.0, 1.5],
    'Second': ['one', 'two', 'three', 'four']
}

# Define the custom row labels (index)
row_labels = ['A', 'B', 'C', 'D']

# Create the DataFrame using the data dictionary and specified index
df = pd.DataFrame(data, index=row_labels)

# Print the created DataFrame to verify
if __name__ == "__main__":
    print(df)
