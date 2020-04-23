# Facial Expression Recognition

Recognize face expressions using a convolutional neural network in Python

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

Firstly, ensure you have Python 3.6 or higher installed

### Installing

Clone this repository

```
git clone https://github.com/SvenKortekaas04/Facial-Expression-Recognition.git
```

Make sure you have the right dependencies installed

```
pip install -r requirements.txt
```

## Prepare Dataset

The dataset used for training the model can be found [here](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data). The dataset consists of 48x48 pixel grayscale images of faces.

The dataset can be preprocessed like below

```
import pandas as pd
import numpy as np

df = pd.read_csv("icml_face_data.csv", skipinitialspace=True)

x_train, y_train, x_test, y_test = [], [], [], []

for i, row in df.iterrows():
    value = row["pixels"].split(" ")

    if "Training" in row["Usage"]:
        x_train.append(np.array(value, dtype="float32"))
        y_train.append(np.array(row["emotion"], dtype="float32"))

    if "PublicTest" in row["Usage"]:
        x_test.append(np.array(value, dtype="float32"))
        y_test.append(np.array(row["emotion"], dtype="float32"))

x_train = np.array(x_train, "float32")
y_train = np.array(y_train, "float32")
x_test = np.array(x_test, "float32")
y_test = np.array(y_test, "float32")
```

## How To Use

For all code examples see the examples folder

### Training a model

```
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
```

### Evaluating the model

```
from model import Model


# Define model
model = Model(num_classes=7, model_name="model_45", model_dir="./")

# Evaluate the trained model
results = model.evaluate(x_test, y_test, batch_size=None, verbose=1)

print(f"Test loss: {results[0]}")
print(f"Test accuracy: {results[1]}")
```

### Making predictions with the model

```
from model import Model
from utils import detect_facial_expressions

import cv2
import numpy as np


# Define model and load weights
model = Model(num_classes=7, model_name="model", model_dir="./")
model.load_weights("model.h5")

# Read the image
image = cv2.imread("assets/images/bill gates.jpg")

# Detect facial expressions
results = detect_facial_expressions(image=image, model=model, cascade_classifier_path="haarcascades/haarcascade_frontalface_alt2.xml")
print(results)
```

## Built With

* [Keras](https://keras.io/)

## Authors

* **Sven Kortekaas** - [SvenKortekaas04](https://github.com/SvenKortekaas04)

## Acknowledgments

* https://medium.com/datadriveninvestor/real-time-facial-expression-recognition-f860dacfeb6a
* https://github.com/opencv/opencv/tree/master/data/haarcascades
