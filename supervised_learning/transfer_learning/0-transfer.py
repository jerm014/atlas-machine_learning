#!/usr/bin/env python3
"""
Train a CNN to classify CIFAR-10 images using transfer learning
"""
from tensorflow import keras as K


def preprocess_data(X, Y):
    """Preprocesses the data for the model."""
    X_p = K.applications.resnet50.preprocess_input(X)
    Y_p = K.utils.to_categorical(Y, 10)
    return X_p, Y_p

# callback for early stopping
class EarlyStoppingAtAccuracy(K.callbacks.Callback):
    """Stop training when the training accuracy reaches a threshold."""
    def __init__(self, accuracy=0.98):
        super(EarlyStoppingAtAccuracy, self).__init__()
        self.accuracy = accuracy

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        acc = logs.get('categorical_accuracy')
        if acc is not None:
            if acc >= self.accuracy:
                print(f"\nReached {self.accuracy*100}% accuracy, stopping.")
                self.model.stop_training = True

if __name__ == "__main__":
    (X_train, Y_train), (X_test, Y_test) = K.datasets.cifar10.load_data()

    # Preprocess the data
    X_train_p, Y_train_p = preprocess_data(X_train.astype('float32'), Y_train)
    X_test_p, Y_test_p = preprocess_data(X_test.astype('float32'), Y_test)

    # Build the base model
    input_shape = (32, 32, 3)
    input_tensor = K.Input(shape=input_shape)
    resized_images = K.layers.Resizing(224, 224)(input_tensor)

    base_model = K.applications.ResNet50(include_top=False,
                                         weights='imagenet',
                                         input_tensor=resized_images)

    for layer in base_model.layers:
        layer.trainable = False  # freeze all base model layers

    x = base_model.output
    x = K.layers.GlobalAveragePooling2D()(x)
    x = K.layers.Dense(256, activation='relu')(x)
    predictions = K.layers.Dense(10, activation='softmax')(x)

    full_model = K.models.Model(inputs=input_tensor, outputs=predictions)

# I think "categorical_accuracy" is the right thing here, not "accuracy".
    full_model.compile(optimizer=K.optimizers.Adam(learning_rate=0.001),
                       loss='categorical_crossentropy',
                       metrics=['categorical_accuracy'])

    early_stopping_callback = EarlyStoppingAtAccuracy(accuracy=0.98)

    full_model.fit(X_train_p,
                   Y_train_p,
                   validation_data=(X_test_p, Y_test_p),
                   epochs=10,
                   batch_size=64,
                   callbacks=[early_stopping_callback])

    # Evaluate the model to ensure metrics are built
    print("\nEvaluating model before saving...")
    test_scores = full_model.evaluate(X_test_p,
                                      Y_test_p,
                                      batch_size=64,
                                      verbose=1)

    print(f"Test loss: {test_scores[0]}")
    print(f"Test accuracy: {test_scores[1]}")

    full_model.save('cifar10.h5')
