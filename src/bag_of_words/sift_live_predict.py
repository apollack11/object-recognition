#!/usr/local/bin/python2.7
import cv2
import numpy as np
from sklearn.externals import joblib
from scipy.cluster.vq import vq, kmeans, whiten

# Load the classifier, class names, scaler, number of clusters and vocabulary
classifier, class_names, std_slr, k, vocabulary = joblib.load("trained_variables.pkl")

cap = cv2.VideoCapture(0)

# Create SIFT object
sift = cv2.SIFT()

# Variables for filtering results
confidences = [0,0,0,0,0]
counter = 0

while(True):
    # Used for filtering results
    if counter > 4:
        counter = 0

    # Capture frame-by-frame
    ret, frame = cap.read()

    # List where all the descriptors are stored
    descriptor_list = []
    kp, des = sift.detectAndCompute(frame, None)
    # frame = cv2.drawKeypoints(frame, kp, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS) # draw SIFT points
    descriptor_list.append(('curFrame', des))

    # Check to make sure descriptor_list has elements
    if descriptor_list[0][1] is not None:
        test_features = np.zeros((1, k), "float32")
        words, distance = vq(whiten(descriptor_list[0][1]), vocabulary)
        for w in words:
            if w >= 0 and w < 100:
                test_features[0][w] += 1

        # Scale the features
        test_features = std_slr.transform(test_features)

        # predict based on classifier (2 objects)
        confidences[counter] = classifier.decision_function(test_features)[0]
        counter = counter + 1

        average_confidence = sum(confidences) / len(confidences)

        # if average_confidence > 0:
        #     prediction = 'soda can'
        # else: # average_confidence < 0:
        #     prediction = 'screwdriver'

        # predictions based on classifier (more than 2)
        predictions = [class_names[i] for i in classifier.predict(test_features)]

        # Display the resulting frame
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, predictions[0], (10,400), font, 2, (255,255,255), 2)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Stop capturing and close
cap.release()
cv2.destroyAllWindows()
