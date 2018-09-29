import cv2
import numpy as np


class FaceDetector:
    def __init__(self, model_file, prototxt, confidence_treshold):
        self.confidence_treshold = confidence_treshold

        # load the model
        print("[INFO] loading face detection model...")
        self.model = cv2.dnn.readNetFromCaffe(prototxt, model_file)

    # input: a numpy matrix of an image
    # output: list of 4-tuples representing the bounding boxes of the detected faces
    def get_detections(self, image):
        # convert the image to a 300x300 normalized blob
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

        # get all detections
        self.model.setInput(blob)
        detections = self.model.forward()

        # filter out the detections with low confidence level, and with non-contained borders
        (h, w) = image.shape[:2]
        filtered_detections = []

        # loop over the detections
        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the
            # prediction
            confidence = detections[0, 0, i, 2]
            if confidence < self.confidence_treshold:
                continue

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            if endX <= w and endY <= h and startX >= 0 and startY >= 0:
                filtered_detections.append((startX, startY, endX, endY))

        return filtered_detections
