"""
Utility functions for facial expression recognition

Written by Sven Kortekaas
"""

from model import Model

import numpy as np
import cv2


def detect_facial_expressions(image, model, cascade_classifier_path: str, score=0.5):
    """
    Detect facial expression in an image

    Args:
        image: The image
        model: The model used to detect the facial expressions in an image
        cascade_classifier_path (str): The path to the cascade classifier file
            used for detecting faces in an image
        score (float): The minimum score the model needs to have when
            predicting facial expressions

    Returns:
        A dictionary containing the box coordinates for the
        detected faces, the class ids and the scores. e.g. {"rois": [], "class_ids": [], "scores": []}
    """

    results = {"rois": [], "class_ids": [], "scores": []}

    # The cascade classifier
    cascade_classifier = cv2.CascadeClassifier(cascade_classifier_path)

    # Grayscale the image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image using the cascade classifier
    faces = cascade_classifier.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)

    for x, y, w, h in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_image = image[y:y + h, x:x + w]

        prediction = model.predict(roi_image, verbose=0)

        # Add the box coordinates, class_id and score to the results list
        results["rois"].append([x, y, w, h])
        results["class_ids"].append(np.argmax(prediction[0]))
        results["scores"].append(float(max(prediction[0])))

    return results
