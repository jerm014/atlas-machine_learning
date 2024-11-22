#!/usr/bin/env python3
import tensorflow as tf
import tensorflow.keras as K
preprocess_data = __import__('0-transfer').preprocess_data

# Print TensorFlow and Keras versions
print(f"TensorFlow version: {tf.__version__}")
print(f"Keras version: {K.__version__}")

# learning_phase isn't in backend any more. it happens automatically.
# K.learning_phase = K.backend.learning_phase
_, (X, Y) = K.datasets.cifar10.load_data()
X_p, Y_p = preprocess_data(X, Y)
model = K.models.load_model('cifar10.h5')

# we're going to get the metrics so we can show everything.
metrics = model.evaluate(X_p, Y_p, batch_size=128, verbose=1)

# Print metrics with labels
metric_names = model.metrics_names
for name, value in zip(metric_names, metrics):
    print(f"{name}: {value}")
