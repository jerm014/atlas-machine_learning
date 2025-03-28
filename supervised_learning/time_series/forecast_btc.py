#!/usr/bin/env python3
"""
Bitcoin price forecasting using RNN architecture.
Uses preprocessed data to train and validate a model for predicting BTC price.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout  # type: ignore
from tensorflow.keras.layers import Bidirectional  # type: ignore
from tensorflow.keras.callbacks import EarlyStopping  # type: ignore
from tensorflow.keras.callbacks import ModelCheckpoint  # type: ignore
from tensorflow.keras.callbacks import ReduceLROnPlateau  # type: ignore
import matplotlib.pyplot as plt
import os
import argparse
import pickle
import time


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train and validate an RNN "
                                     "model for BTC price forecasting.")
    parser.add_argument('--data_dir', type=str, default='preprocessed_data',
                        help='Directory with preprocessed data')
    parser.add_argument('--model_type', type=str, default='lstm',
                        choices=['lstm', 'gru', 'bidirectional'],
                        help='Type of RNN model to use')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--output', type=str, default='model_output',
                        help='Output directory for model and results')
    return parser.parse_args()


def load_data(data_dir):
    """Load preprocessed data."""
    # Load training and validation data
    X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
    y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
    X_val = np.load(os.path.join(data_dir, 'X_val.npy'))
    y_val = np.load(os.path.join(data_dir, 'y_val.npy'))

    # Load scaler for inverse transformation
    with open(os.path.join(data_dir, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)

    # Load feature names
    with open(os.path.join(data_dir, 'features.pkl'), 'rb') as f:
        features = pickle.load(f)

    return X_train, y_train, X_val, y_val, scaler, features


def create_tf_dataset(X, y, batch_size):
    """Create a TensorFlow Dataset for efficient data feeding."""
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    dataset = dataset.shuffle(buffer_size=len(X))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset


def build_model(input_shape, model_type='lstm'):
    """Build the RNN model according to the specified type."""
    model = Sequential()

    if model_type == 'lstm':
        # LSTM-based model
        model.add(LSTM(128, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(0.2))
        model.add(LSTM(64, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1))

    elif model_type == 'gru':
        # GRU-based model
        model.add(GRU(128, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(0.2))
        model.add(GRU(64, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1))

    elif model_type == 'bidirectional':
        # Bidirectional LSTM-based model
        model.add(Bidirectional(
            LSTM(128, return_sequences=True), input_shape=input_shape))
        model.add(Dropout(0.2))
        model.add(Bidirectional(LSTM(64, return_sequences=False)))
        model.add(Dropout(0.2))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1))

    # Compile the model with MSE loss
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def train_model(model, train_dataset, val_dataset, epochs, output_dir):
    """Train the model with early stopping and model checkpointing."""
    # Create callbacks
    callbacks = [
        EarlyStopping(patience=10, monitor='val_loss'),
        ModelCheckpoint(
            filepath=os.path.join(output_dir, 'best_model.keras'),
            monitor='val_loss',
            save_best_only=True
        ),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=5, min_lr=1e-6)
    ]

    # Train the model
    history = model.fit(
        train_dataset,
        epochs=epochs,
        validation_data=val_dataset,
        callbacks=callbacks
    )

    return history


def evaluate_model(model, X_val, y_val, scaler, features):
    """Evaluate the model and calculate relevant metrics."""
    # Make predictions
    predictions = model.predict(X_val)

    # Create a dummy array to inverse transform the predictions
    dummy_array = np.zeros((len(predictions), len(features) + 1))
    dummy_array[:, -1] = predictions.flatten()

    # Inverse transform to get actual price values
    inverse_predictions = scaler.inverse_transform(dummy_array)[:, -1]

    # Do the same for actual values
    dummy_array = np.zeros((len(y_val), len(features) + 1))
    dummy_array[:, -1] = y_val
    actual_values = scaler.inverse_transform(dummy_array)[:, -1]

    # Calculate metrics
    mse = np.mean((inverse_predictions - actual_values)**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(inverse_predictions - actual_values))
    mape = np.mean(
        np.abs((actual_values - inverse_predictions) / actual_values)) * 100

    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'predictions': inverse_predictions,
        'actual': actual_values
    }


def plot_results(history, evaluation_results, output_dir):
    """Plot and save training history and prediction results."""
    # Plot training history
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()

    # Plot predictions vs actual
    plt.subplot(1, 2, 2)
    plt.plot(evaluation_results['actual'][-100:], label='Actual Price')
    plt.plot(evaluation_results['predictions'][-100:],
             label='Predicted Price')
    plt.title('BTC Price Forecast')
    plt.xlabel('Time Step')
    plt.ylabel('Price (USD)')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'results.png'))

    # Save zoomed in view of predictions
    plt.figure(figsize=(12, 6))
    plt.plot(evaluation_results['actual'][-50:],
             label='Actual Price', marker='o')
    plt.plot(evaluation_results['predictions']
             [-50:], label='Predicted Price', marker='x')
    plt.title('BTC Price Forecast (Last 50 Hours)')
    plt.xlabel('Time Step')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'detailed_results.png'))


def main():
    """Main function to train and evaluate the model."""
    args = parse_args()

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Load data
    print("Loading preprocessed data...")
    X_train, y_train, X_val, y_val, scaler, feat = load_data(args.data_dir)
    print(f"Data loaded: {X_train.shape[0]} training samples, "
          f"{X_val.shape[0]} validation samples")

    # Create TensorFlow datasets
    print("Creating TensorFlow datasets...")
    train_dataset = create_tf_dataset(X_train, y_train, args.batch_size)
    val_dataset = create_tf_dataset(X_val, y_val, args.batch_size)

    # Build the model
    print(f"Building {args.model_type} model...")
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_model(input_shape, args.model_type)
    model.summary()

    # Train the model
    print("Training model...")
    start_time = time.time()
    history = train_model(model, train_dataset,
                          val_dataset, args.epochs, args.output)
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")

    # Load the best model
    model = tf.keras.models.load_model(
        os.path.join(args.output, 'best_model.keras'))

    # Evaluate the model
    print("Evaluating model...")
    evaluation_results = evaluate_model(model, X_val, y_val, scaler, feat)

    # Print evaluation metrics
    print(f"Mean Squared Error (MSE): "
          f"{evaluation_results['mse']:.2f}")
    print(f"Root Mean Squared Error (RMSE): "
          f"{evaluation_results['rmse']:.2f}")
    print(f"Mean Absolute Error (MAE): "
          f"{evaluation_results['mae']:.2f}")
    print(f"Mean Absolute Percentage Error (MAPE): "
          f"{evaluation_results['mape']:.2f}%")

    # Plot and save results
    print("Generating plots...")
    plot_results(history, evaluation_results, args.output)

    # Save evaluation metrics
    with open(os.path.join(args.output, 'metrics.txt'), 'w') as f:
        f.write(f"Model Type: {args.model_type}\n")
        f.write(f"Training Time: {training_time:.2f} seconds\n")
        f.write(f"Mean Squared Error (MSE): "
                f"{evaluation_results['mse']:.2f}\n")
        f.write(f"Root Mean Squared Error (RMSE): "
                f"{evaluation_results['rmse']:.2f}\n")
        f.write(f"Mean Absolute Error (MAE): "
                f"{evaluation_results['mae']:.2f}\n")
        f.write(f"Mean Absolute Percentage Error (MAPE): "
                f"{evaluation_results['mape']:.2f}%\n")

    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
