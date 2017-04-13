#!/usr/local/bin/python2.7
import argparse as ap
import cv2
import numpy as np
import os
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from scipy.cluster.vq import vq, kmeans, whiten
import time

def resize_image(img):
    height,width = img.shape[:2]
    aspect_ratio = float(width)/height

    if width > 960:
        width_new = 960
        height_new = int(width_new / aspect_ratio)
        img = cv2.resize(img, (width_new, height_new), interpolation = cv2.INTER_CUBIC)
    return img

# Load the classifier, class names, scaler, number of clusters and vocabulary
# classifier, class_names, std_slr, k, vocabulary = joblib.load("dataset2.pkl")
classifier, class_names, std_slr, k, vocabulary = joblib.load("trained_variables.pkl")

cap = cv2.VideoCapture(0)

# Create SIFT object
sift = cv2.SIFT()

confidences = [0,0,0,0,0]
counter = 0

while(True):
    if counter > 4:
        counter = 0
    # Capture frame-by-frame
    ret, frame = cap.read()

    left_half = frame[0:480, 0:640/2]
    right_half = frame[0:480, 640/2:640]

    # frames = [left_half, right_half]

    # List where all the descriptors are stored
    descriptor_list = []
    kp, des = sift.detectAndCompute(left_half, None)
    # frame = cv2.drawKeypoints(frame, kp, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    descriptor_list.append(('curFrame', des))

    #
    if descriptor_list[0][1] is not None:
        test_features = np.zeros((1, k), "float32")
        words, distance = vq(whiten(descriptor_list[0][1]), vocabulary)
        for w in words:
            if w >= 0 and w < 100:
                test_features[0][w] += 1

        # Scale the features
        test_features = std_slr.transform(test_features)

        # predictions based on classifier (more than 2)
        predictions = [class_names[i] for i in classifier.predict(test_features)]

        # Display the resulting frame
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, predictions[0], (40,40), font, 1, (255,255,255), 2)

    # List where all the descriptors are stored
    descriptor_list = []
    kp, des = sift.detectAndCompute(right_half, None)
    # frame = cv2.drawKeypoints(frame, kp, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    descriptor_list.append(('curFrame', des))

    #
    if descriptor_list[0][1] is not None:
        test_features = np.zeros((1, k), "float32")
        words, distance = vq(whiten(descriptor_list[0][1]), vocabulary)
        for w in words:
            if w >= 0 and w < 100:
                test_features[0][w] += 1

        # Scale the features
        test_features = std_slr.transform(test_features)

        # predictions based on classifier (more than 2)
        predictions = [class_names[i] for i in classifier.predict(test_features)]

        # Display the resulting frame
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, predictions[0], (360,40), font, 1, (255,255,255), 2)

    cv2.line(frame, (318,0), (318,480), (255,0,0), 4)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
