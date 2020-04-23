"""
Facial expression recognition model

Written by Sven Kortekaas
"""

import numpy as np
import cv2
import os

from keras.models import Sequential
import keras.layers as KL
import keras


class Model:
    def __init__(self, num_classes: int, model_name='model', model_dir="./"):
        """
        Facial expression recognition model

        Args:
            num_classes (int): The number of classes
            model_name (str, optional): The name of the model. Useful when you experiment with different models
            model_dir (str): Directory to save training logs and trained weights
        """

        self.num_classes = num_classes
        self.model_name = model_name
        self.model_dir = model_dir

        # Build the model
        self.model = self.build(num_classes=num_classes)

    def build(self, num_classes):
        """
        Build the facial expression recognition model

        Args:
            num_classes (int): The number of classes

        Returns:
            The created model
        """

        # Define model
        model = Sequential()

        # Add layers to the model
        model.add(KL.Conv2D(64, (3, 3), strides=(1, 1), activation="relu", input_shape=(48, 48, 1)))
        model.add(KL.Conv2D(64, (3, 3), strides=(1, 1), activation="relu"))
        model.add(KL.BatchNormalization())
        model.add(KL.MaxPooling2D(pool_size=(2, 2)))
        model.add(KL.Dropout(0.4))

        model.add(KL.Conv2D(64, (3, 3), strides=(1, 1), activation="relu"))
        model.add(KL.Conv2D(64, (3, 3), strides=(1, 1), activation="relu"))
        model.add(KL.BatchNormalization())
        model.add(KL.MaxPooling2D(pool_size=(2, 2)))
        model.add(KL.Dropout(0.4))

        model.add(KL.Conv2D(128, (3, 3), strides=(1, 1), activation="relu"))
        model.add(KL.Conv2D(128, (3, 3), strides=(1, 1), activation="relu"))
        model.add(KL.BatchNormalization())
        model.add(KL.MaxPooling2D(pool_size=(2, 2)))
        model.add(KL.Dropout(0.4))

        model.add(KL.Flatten())

        model.add(KL.Dense(1024, activation="relu"))
        model.add(KL.Dropout(0.4))
        model.add(KL.Dense(1024, activation="relu"))
        model.add(KL.Dropout(0.4))
        model.add(KL.Dense(num_classes, activation="softmax"))

        # Compile the model
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        return model

    def load_weights(self, file: str, by_name=False):
        """
        Load weights on to the model

        Args:
            file (str): The path to the weights file
            by_name (bool): True if the model has a different architecture (with some layers in common)
        """

        self.model.load_weights(file, by_name=by_name)

    def compile(self, optimizer: str, loss: str, metrics: list):
        """
        Configure the learning process of the model for training

        Args:
            optimizer (str): This could be the string identifier of an existing optimizer (such as adam)
            loss (str): This is the objective that the model will try to minimize
            metrics (list): A list of metrics
        """

        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def evaluate(self, x, y, batch_size=None, verbose=1):
        """
        Evaluate the trained model

        Args:
            x: Input data
            y: Target data
            batch_size (int or None): The number of training examples utilized in one iteration
            verbose (int): By setting verbose 0 or 1 you can specify
                how you want the training progress for each epoch to be displayed.
                - 0: Nothing is shown
                - 1: Will show an animated progress bar

        Returns:
            Returns the loss value & metrics values for the model in test mode
        """

        results = self.model.evaluate(x=x, y=y, batch_size=batch_size, verbose=verbose)

        return results

    def train(self, x_train, y_train, x_test, y_test, epochs=10, batch_size=32, verbose=1, custom_callbacks=None, shuffle=True):
        """
        Train the model

        Args:
            x_train, y_train: The training data for the model
            x_test, y_test: The validation data for the model
            epochs (int): The number of passes through the entire training dataset
            batch_size (int): The number of training examples utilized in one iteration
            verbose (int): By setting verbose 0, 1 or 2 you can specify
                how you want the training progress for each epoch to be displayed.
                - 0: Nothing is shown
                - 1: Will show an animated progress bar
                - 2: Will just mention the number of the epoch
            custom_callbacks (list): Custom callbacks to be called when training the model
            shuffle (bool): Whether to shuffle the training data before each epoch
        """

        # Create model directory if it doesn't already exist
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        # Callbacks
        callbacks = [
                        keras.callbacks.ModelCheckpoint(self.model_dir + self.model_name + ".h5", verbose=0, save_weights_only=True)
                    ]

        # Add custom callbacks
        if custom_callbacks:
            callbacks += custom_callbacks

        # Normalize pixel values between 0 and 1
        x_train /= 255
        x_test /= 255

        # Reshape to be [samples][width][height][channels]
        x_train = x_train.reshape(x_train.shape[0], 48, 48, 1)
        x_test = x_test.reshape(x_test.shape[0], 48, 48, 1)

        # One hot encoded outputs
        y_train = keras.utils.to_categorical(y_train, num_classes=self.num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes=self.num_classes)

        # Train the model
        self.model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=verbose,
                        callbacks=callbacks,
                        validation_data=(x_test, y_test),
                        shuffle=shuffle)

    def predict(self, x, batch_size=None, verbose=0):
        """
        Make predictions with the model

        Args:
            x: Input data
            batch_size (int): The number of training examples utilized in one iteration
            verbose (int): By setting verbose 0, 1 or 2 you can specify
                how you want the training progress for each epoch to be displayed.
                - 0: Nothing is shown
                - 1: Will show an animated progress bar
                - 2: Will just mention the number of the epoch

        Returns:
            The model's prediction
        """

        # Resize the image to 48x48 pixels
        x = cv2.resize(x, (48, 48))

        # Grayscale the image
        gray = np.dot(x[...,:3], [0.299, 0.587, 0.114])

        # Reshape the image
        gray = gray.reshape(1, 48, 48, 1)

        # Normalize pixel values between 0 and 1
        gray /= 255

        # Make prediction with the model
        prediction = self.model.predict(gray, batch_size=batch_size, verbose=verbose)

        return prediction
