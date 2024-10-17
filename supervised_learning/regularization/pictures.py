

from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(
    monitor='val_loss',  # Monitor validation loss
    patience=10,  # Number of epochs with no improvement after which training
                  # will be stopped
    restore_best_weights=True  # Restore model weights from the epoch with
                               # the best value of the monitored quantity
)

model.fit(X_train, y_train, validation_data=(X_val, y_val), callbacks=[early_stopping])


from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=20,  # Rotate the image slightly
    width_shift_range=0.1,  # Shift the image horizontally
    height_shift_range=0.1,  # Shift the image vertically
    horizontal_flip=True  # Flip the image horizontally
)

# Fit the augmentation to your data
datagen.fit(X_train)



from tensorflow.keras.layers import Dropout

model = Sequential([
    Dense(64, activation='relu', input_shape=(input_dim,)),
    Dropout(0.5),  # Randomly "remove" 50% of the neurons in this layer during training
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])


def lasso_regression(X, y, alpha, n_iterations=1000, learning_rate=0.01):
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    
    for _ in range(n_iterations):
        # Compute predictions
        y_pred = np.dot(X, weights)
        
        # Compute gradients
        dw = np.dot(X.T, (y_pred - y)) / n_samples
        
        # Update weights with L1 regularization
        for i in range(n_features):
            if weights[i] > 0:
                weights[i] = max(0, weights[i] - learning_rate * (dw[i] + alpha))
            else:
                weights[i] = min(0, weights[i] - learning_rate * (dw[i] - alpha))
    
    return weights

    