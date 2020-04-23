"""
Example on how to make predictions with a
facial expression recognition model

Written by Sven Kortekaas
"""

from model import Model
from utils import detect_facial_expressions

import cv2
import numpy as np


# The 7 different emotions the model can detect
emotions = ("Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral")

# Define model and load weights
model = Model(num_classes=7, model_name="model", model_dir="./")
model.load_weights("model.h5")

# Read the image
image = cv2.imread("assets/images/bill gates.jpg")

# Detect facial expressions
results = detect_facial_expressions(image=image, model=model, cascade_classifier_path="haarcascades/haarcascade_frontalface_alt2.xml")

for i in range(len(results["rois"])):
    # Get the x, y, w, h values for the region of interest
    x, y, w, h = results["rois"][i]

    # Get the corresponding emotion and score
    emotion = emotions[results["class_ids"][i]]
    score = results["scores"][i]

    # Create a label
    label = "{} {:.0f}%".format(emotion, score * 100)

    # Draw bounding box and label around detected object
    cv2.rectangle(image, (x, y), (x + w, y + h), color=[255, 0, 0], thickness=2)
    cv2.putText(image, label, (x, y - 7), cv2.FONT_HERSHEY_PLAIN, fontScale=0.75, color=[255, 0, 0], thickness=1)

# Display the image
cv2.imshow("Image", image)
cv2.waitKey(0)
