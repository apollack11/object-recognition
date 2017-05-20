#!/usr/local/bin/python2.7
import cv2
import numpy as np
from sklearn.externals import joblib
from scipy.cluster.vq import vq, kmeans, whiten

# Load the classifier, class names, scaler, number of clusters and vocabulary
classifier, class_names, std_slr, k, vocabulary = joblib.load("dataset6.pkl")
# classifier, class_names, std_slr, k, vocabulary = joblib.load("trained_variables.pkl")

cap = cv2.VideoCapture(0)

# Create SIFT object
sift = cv2.SIFT()

# Variables for filtering results
confidences = [0,0,0,0,0]
counter = 0

needResizing = False

while(True):
    # Used for filtering results
    if counter > 4:
        counter = 0

    # Capture frame-by-frame
    ret, frame = cap.read()

    # Split frame into left and right halves
    left_frame = frame[0:480, 0:640/3]
    middle_frame = frame[0:480, 640/3:2*640/3]
    right_frame = frame[0:480, 2*640/3:640]

    # Create list of left and right images
    frames = [left_frame, middle_frame, right_frame]

    # Check to make sure descriptor_list has elements
    items = []
    for i,f in enumerate(frames):
        kps, des = sift.detectAndCompute(f, None)
        frames[i] = cv2.drawKeypoints(f, kps, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS) # draw SIFT points

        if des is not None and len(kps) > 15:
            items.append([kp for kp in kps])
            test_features = np.zeros((1, k), "float32")
            words, distance = vq(whiten(des), vocabulary)
            for w in words:
                if w >= 0 and w < 100:
                    test_features[0][w] += 1

            # Scale the features
            test_features = std_slr.transform(test_features)

            # predictions based on classifier (more than 2)
            predictions = [class_names[j] for j in classifier.predict(test_features)]

            # Add label to half of the image
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frames[i], predictions[0], (40,100), font, 1, (255,255,255), 2)

            gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)

            ones = [0,0,0]
            ones[i] = 1

            contours, hierarchy = cv2.findContours(gray,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key = cv2.contourArea, reverse = True)[:10]
            cnt = contours[0]
            rect = cv2.minAreaRect(cnt)
            box = np.int0(cv2.cv.BoxPoints(rect))
            cv2.drawContours(frames[i],[box],0,(255*ones[0],255*ones[1],255*ones[2]),2)


    for i,item in enumerate(items):
        xsum = 0
        ysum = 0
        for kp in item:
            xsum += kp.pt[0]
            ysum += kp.pt[1]
        avg_item_pt = (xsum/len(item), ysum/len(item))
        ones = [0,0,0]
        ones[i] = 1
        cv2.circle(frames[i],(int(avg_item_pt[0]),int(avg_item_pt[1])), 5, (255*ones[0],255*ones[1],255*ones[2]), -1)

    # Combine smaller frames into one
    frame[0:480, 0:640/3] = frames[0]
    frame[0:480, 640/3:2*640/3] = frames[1]
    frame[0:480, 2*640/3:640] = frames[2]

    # Add a dividing line down the middle of the frame
    cv2.line(frame, (640/3,0), (640/3,480), (255,0,0), 4)
    cv2.line(frame, (2*640/3,0), (2*640/3,480), (255,0,0), 4)

    # Resize image to fit monitor (if monitor is attached)
    if needResizing:
        frame = cv2.resize(frame, (1920, 1080), interpolation = cv2.INTER_CUBIC)

    # Display the resulting frame
    cv2.imshow('f', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Stop capturing and close
cap.release()
cv2.destroyAllWindows()
