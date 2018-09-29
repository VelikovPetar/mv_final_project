from imutils.video import VideoStream
from detection import FaceDetector
from classification import EmotionClassifier
from models import big_XCEPTION
import imutils
import cv2
import time

classes = ['Fear', 'Disgust', 'Surprise', 'Contempt', 'Anger', 'Neutral', 'Happiness', 'Sadness']

# initialize all components
face_detector = FaceDetector(
    model_file='res10_300x300_ssd_iter_140000.caffemodel',
    prototxt='deploy.prototxt',
    confidence_treshold=0.5)

print("[INFO] loading the classifier model...")
classifier_model = big_XCEPTION(input_shape=(64, 64, 1), num_classes=8)
classifier_model.load_weights('big_XCEPTION_300-epochs_16-batch_size.h5')

emotion_classifier = EmotionClassifier(
    shape_predictor='shape_predictor_68_face_landmarks.dat',
    image_size=64,
    classifier_model=classifier_model)

# initialize the video stream and allow the cammera sensor to warmup
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2)

# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 400 pixels
    frame = vs.read()
    frame = imutils.resize(frame, width=400)
    frame = cv2.flip(frame, 1)

    detections = face_detector.get_detections(frame)

    for (startX, startY, endX, endY) in detections:
        # do an emotion classification of the face
        #classification_result = classify_emotion(frame, startX, startY, endX, endY)

        prediction = emotion_classifier.classify(frame, startX, startY, endX, endY)

        prediction_pairs = []
        for i in range(8):
            prediction_pairs.append((classes[i], prediction[0][i]))

        # sort the pairs in decreasing order
        prediction_pairs = sorted(prediction_pairs, key=lambda x: x[1], reverse=True)

        # draw the bounding box of the face
        cv2.rectangle(frame, (startX, startY), (endX, endY), (20, 220, 0), 2)

        # write the predictions
        cv2.putText(frame, '%.1f%% %s' % (prediction_pairs[0][1] * 100, prediction_pairs[0][0]), (endX + 5, startY + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)

        y_offset = 25
        for i in range(1, 8):
            cv2.putText(frame, '%.1f%% %s' % (prediction_pairs[i][1] * 100, prediction_pairs[i][0]), (endX + 5, startY + y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
            y_offset += 15


    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()