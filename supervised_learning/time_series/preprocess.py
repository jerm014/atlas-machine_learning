#!/usr/bin/env python3
"""
Preprocessing script for Bitcoin price data from Coinbase and Bitstamp.
This script prepares the data for use in RNN-based price forecasting, yo.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os
import argparse
import pickle

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Preprocess Bitcoin price data for RNN forecasting.")
    parser.add_argument('--coinbase', type=str, required=True,
        help='Path to Coinbase dataset CSV file')
    parser.add_argument('--bitstamp', type=str, required=True,
        help='Path to Bitstamp dataset CSV file')
    parser.add_argument('--output', type=str, default='preprocessed_data',
        help='Output directory')
    return parser.parse_args()

def load_data(coinbase_path, bitstamp_path):
    """Load datasets from both exchanges and merge them."""
    # Column names for the datasets
    columns = ['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume_(BTC)',
        'Volume_(Currency)', 'Weighted_Price']
    
    # Check if files have headers by examining the first few lines
    with open(coinbase_path, 'r') as f:
        first_line = f.readline().strip()
    
    # If first line looks like a header, use header=0, otherwise use names
    # with header=None
    has_header = first_line.startswith('Timestamp') or ',' in first_line and not \
        first_line.split(',')[0].isdigit()
    
    # Load datasets
    if has_header:
        # Files have headers, so use them directly
        coinbase_df = pd.read_csv(coinbase_path, low_memory=False)
        bitstamp_df = pd.read_csv(bitstamp_path, low_memory=False)
    else:
        # Files don't have headers, so provide our own names
        coinbase_df = pd.read_csv(coinbase_path, names=columns, header=None,
            low_memory=False)
        bitstamp_df = pd.read_csv(bitstamp_path, names=columns, header=None,
            low_memory=False)
    
    # Add source column to identify the exchange
    coinbase_df['source'] = 'coinbase'
    bitstamp_df['source'] = 'bitstamp'
    
    # Merge datasets
    df = pd.concat([coinbase_df, bitstamp_df], ignore_index=True)
    
    # Convert timestamp to datetime
    # Ensure unix_timestamp column is numeric
    df['Timestamp'] = pd.to_numeric(df['Timestamp'],
        errors='coerce')
    
    # Drop rows where timestamp couldn't be converted to numeric
    df = df.dropna(subset=['Timestamp'])
    
    # Convert to datetime
    df['datetime'] = pd.to_datetime(df['Timestamp'], unit='s')
    
    # Sort by datetime
    df = df.sort_values('datetime')
    
    return df

def clean_data(df):
    """Clean and filter the data."""
    # Convert numeric columns to proper numeric types
    numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume_(BTC)',
        'Volume_(Currency)', 'Weighted_Price']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop rows with missing values
    df = df.dropna()
    
    # Filter out rows with zero or unrealistic values
    df = df[(df['Close'] > 0) & (df['High'] > 0) & (df['Low'] > 0)]
    
    # Keep only relevant features
    relevant_features = ['datetime', 'Timestamp', 'Open', 'High', 'Low',
        'Close', 'Volume_(BTC)', 'Weighted_Price']
    df = df[relevant_features]
    
    return df

def aggregate_hourly(df):
    """Aggregate data to hourly intervals."""
    # Use the source with most data for duplicate timestamps
    df = df.drop_duplicates(subset=['Timestamp'], keep='first')
    
    # Set datetime as index for resampling
    df = df.set_index('datetime')
    
    # Resample to hourly data
    hourly_df = df.resample('1h').agg({
        'Timestamp': 'first',
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume_(BTC)': 'sum',
        'Weighted_Price': 'mean'
    })
    
    # Reset index to get datetime as a column again
    hourly_df = hourly_df.reset_index()
    
    # Drop rows with missing values after resampling
    hourly_df = hourly_df.dropna()
    
    return hourly_df

def create_sequences(df, sequence_length=24, forecast_hours=1):
    """Create sequences for RNN training."""
    # Selected features for the model
    features = ['Open', 'High', 'Low', 'Close', 'Volume_(BTC)',
        'Weighted_Price']
    
    # Extract the target variable (close price after forecast_hours)
    target_column = df['Close'].shift(-forecast_hours)
    
    # Create a dataframe with features and target
    data = df[features].copy()
    data['target'] = target_column
    
    # Drop rows with NaN values
    data = data.dropna()
    
    # Normalize the data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    
    # Create sequences
    X = []
    y = []
    
    for i in range(len(scaled_data) - sequence_length):
        # All features except target
        X.append(scaled_data[i:i + sequence_length, :-1])
        # Only target
        y.append(scaled_data[i + sequence_length, -1])

    X = np.array(X)
    y = np.array(y)

    # Split into training and validation sets (80% train, 20% validation)
    train_size = int(len(X) * 0.8)
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]
    
    return X_train, y_train, X_val, y_val, scaler, features

def main():
    """Main preprocessing function."""
    args = parse_args()
    
    # Load and merge data
    print("Loading data...")
    df = load_data(args.coinbase, args.bitstamp)
    print(f"Loaded data with {len(df)} entries")
    
    # Clean data
    print("Cleaning data...")
    cleaned_df = clean_data(df)
    print(f"Cleaned data has {len(cleaned_df)} entries")
    
    # Aggregate to hourly data
    print("Aggregating to hourly data...")
    hourly_df = aggregate_hourly(cleaned_df)
    print(f"Hourly data has {len(hourly_df)} entries")
    
    # Create sequences for RNN
    print("Creating sequences for RNN training...")
    X_train, y_train, X_val, y_val, scaler, feat = create_sequences(hourly_df)
    print(f"Created {len(X_train)} training sequences and {len(X_val)} valida"
        "tion sequences")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    # Save preprocessed data
    print("Saving preprocessed data...")
    np.save(os.path.join(args.output, 'X_train.npy'), X_train)
    np.save(os.path.join(args.output, 'y_train.npy'), y_train)
    np.save(os.path.join(args.output, 'X_val.npy'), X_val)
    np.save(os.path.join(args.output, 'y_val.npy'), y_val)
    
    # Save scaler and features for later use
    with open(os.path.join(args.output, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    
    # Save feature names
    with open(os.path.join(args.output, 'features.pkl'), 'wb') as f:
        pickle.dump(feat, f)
    
    print(f"Preprocessing complete. Data saved to {args.output}")

if __name__ == "__main__":
    main()