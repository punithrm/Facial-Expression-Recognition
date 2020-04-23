"""
Example on how to train a facial expression recognition model

Written by Sven Kortekaas
"""

from model import Model


# Define model
model = Model(num_classes=7, model_name="model_45", model_dir="./")

# Train the model
model.train(x_train, y_train,
            x_test, y_test,
            epochs=10,
            batch_size=32,
            verbose=1,
            custom_callbacks=None,
            shuffle=True)

# Evaluate the trained model
results = model.evaluate(x_test, y_test, batch_size=None, verbose=1)

print(f"Test loss: {results[0]}")
print(f"Test accuracy: {results[1]}")
