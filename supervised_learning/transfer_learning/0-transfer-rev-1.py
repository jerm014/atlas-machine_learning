#!/usr/bin/env python3
"""
Train a CNN to classify CIFAR-10 images using transfer learning
"""
from tensorflow import keras as K


def preprocess_data(X, Y):
    """Preprocesses the data for the model"""
    X_p = X / 255.0
    Y_p = K.utils.to_categorical(Y, 10)
    return X_p, Y_p

if __name__ == "__main__":
    (X_train, Y_train), (X_test, Y_test) = K.datasets.cifar10.load_data()

    # Preprocess the data
    X_train_p, Y_train_p = preprocess_data(X_train.astype('float32'), Y_train)
    X_test_p, Y_test_p = preprocess_data(X_test.astype('float32'), Y_test)

    # Build the base model
    input_shape = (32, 32, 3)
    input_tensor = K.Input(shape=input_shape)
    resized_images = K.layers.UpSampling2D(size=(7, 7))(input_tensor)

    base_model = K.applications.ResNet50(include_top=False, 
                                       weights='imagenet',
                                       input_tensor=resized_images)

    x = base_model.output
    x = K.layers.Flatten()(x)
    x = K.layers.Dense(128, activation='relu')(x)
    predictions = K.layers.Dense(10, activation='softmax')(x)

    full_model = K.models.Model(inputs=input_tensor, outputs=predictions)

    full_model.compile(optimizer=K.optimizers.Adam(),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

    # Train the model
    full_model.fit(X_train_p, Y_train_p,
                  validation_data=(X_test_p, Y_test_p),
                  epochs=5, batch_size=32)

    # Save the model
    full_model.save('cifar10.h5')
