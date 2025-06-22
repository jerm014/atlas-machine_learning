#!/usr/bin/env python3

import matplotlib.pyplot as plt
import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

# 1. Remove the Weighted_Price column
if 'Weighted_Price' in df.columns:
    df.drop(columns=['Weighted_Price'], inplace=True)

# 2. Rename the column Timestamp to Date
df.rename(columns={'Timestamp': 'Date'}, inplace=True)

# 3. Convert the timestamp values to date values
# Assuming timestamps are in seconds (Unix epoch)
df['Date'] = pd.to_datetime(df['Date'], unit='s')

# 4. Index the data frame on Date
df.set_index('Date', inplace=True)

# 5. Missing values in Close should be set to the previous row value
df['Close'].fillna(method='ffill', inplace=True)

# 6. Missing values in High, Low, Open should be set to the same row Close
# Ensure 'Close' column has no NaNs before using it for filling other columns
for col in ['High', 'Low', 'Open']:
    if col in df.columns:
        df[col].fillna(df['Close'], inplace=True)

# 7. Missing values in Volume_(BTC) and Volume_(Currency) should be set to 0
for col in ['Volume_(BTC)', 'Volume_(Currency)']:
    if col in df.columns:
        df[col].fillna(0, inplace=True)

# 8. Plot the data from 2017 and beyond at daily intervals
# Filter data from 2017 onwards
df_2017_onwards = df.loc['2017-01-01':]

# Group the values of the same day such that:
# High: max, Low: min, Open: mean, Close: mean
# Volume(BTC): sum, Volume(Currency): sum
df_daily = df_2017_onwards.resample('D').agg({
    'High': 'max',
    'Low': 'min',
    'Open': 'mean',
    'Close': 'mean',
    'Volume_(BTC)': 'sum',
    'Volume_(Currency)': 'sum'
})

# Return the transformed pd.DataFrame before plotting
transformed_df = df_daily

# Plotting the transformed data
plt.figure(figsize=(12, 8))
transformed_df.plot(subplots=True, figsize=(12, 10))
plt.suptitle('Daily Bitcoin Price and Volume Data (2017 Onwards)')
plt.tight_layout(rect=[0, 0.03, 1, 0.96])  # Adjust layout to prevent overlap
plt.show()
