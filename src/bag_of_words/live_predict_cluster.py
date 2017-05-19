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

needResizing = False

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # List where all the descriptors are stored
    kp, des = sift.detectAndCompute(frame, None)
    # frame = cv2.drawKeypoints(frame, kp, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS) # draw SIFT points

    pts = [keypoint.pt for keypoint in kp]
    item_pts = [[pt for pt in pts if pt[0] < 213],[pt for pt in pts if pt[0] > 213 and pt[0] < 426],[pt for pt in pts if pt[0] > 426]]

    for i,item in enumerate(item_pts):
        xsum = 0
        ysum = 0
        for pt in item:
            xsum += pt[0]
            ysum += pt[1]
        avg_item_pt = (xsum/len(item), ysum/len(item))
        ones = [0,0,0]
        ones[i] = 1
        cv2.circle(frame,(int(avg_item_pt[0]),int(avg_item_pt[1])), 5, (255*ones[0],255*ones[1],255*ones[2]), -1)

    # Check to make sure des has elements and there are more than 15 keypoints
    if des is not None and len(kp) > 15:
        test_features = np.zeros((1, k), "float32")
        words, distance = vq(whiten(des), vocabulary)
        for w in words:
            if w >= 0 and w < 100:
                test_features[0][w] += 1

        # Scale the features
        test_features = std_slr.transform(test_features)

        # predictions based on classifier (more than 2)
        predictions = [class_names[i] for i in classifier.predict(test_features)]

        # Add label to half of the image
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, predictions[0], (225,100), font, 1, (255,255,255), 2)

    if needResizing:
        frame = cv2.resize(frame, (1920, 1080), interpolation = cv2.INTER_CUBIC)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Stop capturing and close
cap.release()
cv2.destroyAllWindows()
