"""
Example on how to make predictions with a
facial expression recognition model

Written by Sven Kortekaas
"""

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
