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
classifier, class_names, std_slr, k, vocabulary = joblib.load("trained_variables.pkl")

cap = cv2.VideoCapture(0)

# Create SIFT object
sift = cv2.SIFT()

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # List where all the descriptors are stored
    descriptor_list = []
    kp, des = sift.detectAndCompute(frame, None)
    frame = cv2.drawKeypoints(frame, kp, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    descriptor_list.append(('curFrame', des))

    descriptors = descriptor_list[0][1]
    for image_name, descriptor in descriptor_list[1:]:
        if descriptor is not None:
            descriptors = np.vstack((descriptors, descriptor))
    #
    if descriptors is not None:
        test_features = np.zeros((1, k), "float32")
        words, distance = vq(descriptor_list[0][1], vocabulary)
        for w in words:
            test_features[0][w] += 1


        # Perform Tf-Idf vectorization
        nbr_occurences = np.sum( (test_features > 0) * 1, axis = 0)
        idf = np.array(np.log((1.0 * 1 + 1) / (1.0 * nbr_occurences + 1)), 'float32')

        # Scale the features
        test_features = std_slr.transform(test_features)

        # Perform the predictions
        predictions = [class_names[i] for i in classifier.predict(test_features)]

        # Our operations on the frame come here
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Display the resulting frame
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, predictions[0], (10,400), font, 2, (255,255,255), 2)

    # frame = cv2.resize(frame, (1920/2, 1080/2), interpolation = cv2.INTER_CUBIC)

    cv2.imshow('frame', frame)
    # cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('frame', 1920/2, 1080/2)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
