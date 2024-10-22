

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

  def early_stopping(cost, opt_cost, threshold, patience, count):
    if cost < opt_cost - threshold:
        count = 0
    else:
        count += 1

    return count >= patience, count


    @tf.function
    def add_noise(x):
        noise = tf.random.normal(shape=tf.shape(x), mean=0.0, stddev=0.1)
        return x + noise

    @tf.function
    def random_scale(x):
        scale = tf.random.uniform(shape=(), minval=0.8, maxval=1.2)
        return x * scale

    @tf.function
    def random_rotate(x):
        angle = tf.random.uniform(shape=(), minval=-np.pi/4, maxval=np.pi/4)
        rotation_matrix = tf.stack([
            [tf.cos(angle), -tf.sin(angle)],
            [tf.sin(angle), tf.cos(angle)]
        ])
        return tf.matmul(tf.reshape(x, [-1, 2]), rotation_matrix)



def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    m = Y.shape[1]
    dZ = cache[f'A{L}'] - Y

    for layer in reversed(range(1, L + 1)):
        A_prev = cache[f'A{layer-1}']
        W = weights[f'W{layer}']
        b = weights[f'b{layer}']

        dW = (1 / m) * np.dot(dZ, A_prev.T)
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

        if layer > 1:
            dA_prev = np.dot(W.T, dZ) # Compute gradient w.r.t. previous
                                      # layers activation (backpropagation step)
            dA_prev *= cache[f'D{layer-1}']  # Apply dropout mask
            dA_prev /= keep_prob  # Scale the values
            dZ = dA_prev * (1 - np.power(A_prev, 2))  # Derivative of tanh

        # Update weights and biases
        weights[f'W{layer}'] -= alpha * dW
        weights[f'b{layer}'] -= alpha * db

