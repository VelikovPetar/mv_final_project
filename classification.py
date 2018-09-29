from imutils.face_utils import FaceAligner
import numpy as np
import cv2
import dlib


class EmotionClassifier:
    def __init__(self, shape_predictor, image_size, classifier_model):
        # create the facial landmark predictor and the face aligner
        predictor = dlib.shape_predictor(shape_predictor)
        self.face_aligner = FaceAligner(predictor, desiredFaceWidth=image_size)
        self.classifier_model = classifier_model


    def classify(self, image, start_x, start_y, end_x, end_y):
        # preprocessing: allign, scale and crop the face in the image
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        aligned_face = self.face_aligner.align(grayscale, grayscale, dlib.rectangle(start_x, start_y, end_x, end_y))

        # convert input image in 3-dimensional vector
        input_image = np.atleast_3d(aligned_face)

        # make the prediction
        prediction = self.classifier_model.predict(np.expand_dims(input_image, axis=0))

        return prediction


