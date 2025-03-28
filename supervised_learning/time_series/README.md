# Bitcoin Price Forecasting using RNNs

This project implements a machine learning system that uses Recurrent Neural Networks (RNNs) to forecast Bitcoin prices based on historical data from Coinbase and Bitstamp exchanges.

## Overview

The system takes advantage of RNN architectures to analyze past 24 hours of Bitcoin trading data and predict the price at the close of the next hour. Three different RNN architectures developed in the previous project are supported:
- LSTM (Long Short-Term Memory)
- GRU (Gated Recurrent Unit)
- Bidirectional LSTM

## Requirements

* Python: (tested with 3.9.13*)
* TensorFlow: (tested with 2.15.0)
* NumPy: (tested with 1.26.4)
* Pandas: (tested with 2.2.3)
* Scikit-learn: (tested with 1.5.0)
* Matplotlib: (tested with 3.9.4)

_note * - additionally, the Python version in use is MSC v.1929 64 bit (AMD64)_

 
You can install all requirements using:
```
pip install tensorflow==2.15.0 numpy==1.26.4 pandas==2.2.3 scikit-learn==1.5.0 matplotlib==3.9.4
```

## Data

The system uses Bitcoin price data from Coinbase and Bitstamp exchanges. Each dataset contains minute-by-minute price information with the following columns:
* Unix timestamp ["Timestamp"]
* Open price (USD) ["Open"]
* High price (USD) ["High"]
* Low price (USD) ["Low"]
* Close price (USD) ["Close"]
* BTC volume ["Volume_(BTC)"]
* USD volume ["Volume_(USD)"]
* Volume-weighted average price (USD) ["Weighted_Price"]

## Files

* `preprocess_data.py`: Script for cleaning, aggregating, and preparing the data for model training
* `forecast_btc.py`: Script for building, training, and evaluating RNN models
* `README.md`: You're reading it

## Preprocessing

The preprocessing script (`preprocess_data.py`) does 6 things:

1. Loads and merges data from both exchanges
2. Cleans the data by removing missing and unrealistic values
3. Aggregates the data to hourly intervals (from minute-by-minute)
4. Creates sequences of 24-hour windows for the RNN training
5. Normalizes the data to optimize model training
6. Splits the data into training and validation sets

### Usage

```
python preprocess_data.py --coinbase path/to/coinbase_data.csv --bitstamp path/to/bitstamp_data.csv --output preprocessed_data
```

Arguments:
- `--coinbase`: Path to Coinbase dataset CSV file (required)
- `--bitstamp`: Path to Bitstamp dataset CSV file (required)
- `--output`: Output directory for preprocessed data (default: "preprocessed_data")

## Model Training and Evaluation

The forecast script (`forecast_btc.py`) also does 6 things:

1. Loads the preprocessed data
2. Creates TensorFlow datasets for efficient training
3. Builds the selected RNN architecture
4. Trains the model with early stopping and checkpointing
5. Evaluates the model on validation data
6. Generates visualizations of results

### Usage

```
python forecast_btc.py --data_dir preprocessed_data --model_type lstm --epochs 50 --batch_size 32 --output model_output
```

```
Arguments:
  --data_dir: Directory with preprocessed data (default: "preprocessed_data")
  --model_type: Type of RNN model to use (choices: "lstm" (default), "gru", "bidirectional")
  --epochs: Number of training epochs (default: 50)
  --batch_size: Batch size for training (default: 32)
  --output: Output directory for model and results (default: "model_output")
```

## Output

The forecast script produces:

* A trained model saved in keras format
* Plots of training/validation loss
* Plots of predicted vs actual BTC prices
* A text file with evaluation metrics:
   - Mean Squared Error (MSE)
   - Root Mean Squared Error (RMSE)
   - Mean Absolute Error (MAE)
   - Mean Absolute Percentage Error (MAPE)

## Example Workflow

First, preprocess the raw data:
```
python preprocess_data.py --coinbase data/coinbase.csv --bitstamp data/bitstamp.csv --output processed_btc_data
```

Train and evalate a bidirectional LSTM model:
```
python forecast_btc.py --data_dir processed_btc_data --model_type bidirectional --epochs 100 --batch_size 64 --output bidirectional_results
```

Try different architectures and compare results:
```
python forecast_btc.py --data_dir processed_btc_data --model_type gru --output gru_results
```

## Notes on Model Performance

* The forecasting task is challenging due to the volatile nature of cryptocurrency prices. Like, really?
* Performance may vary depending on market conditions in the training/validation period.
* Bidirectional models often perform better but require more computational resources.
* Consider external factors (market news, regulations, etc.) that are not captured in the price data alone.

## Future Improvements

Potential enhancements to the system could include:
* Adding more features from external sources (e.g., market sentiment, transaction volume across all exchanges)
* Implementing attention mechanisms to focus on the most relevant time periods
* Extending the forecast window to predict multiple hours ahead
* Incorporating traditional time series analysis techniques alongside deep learning
