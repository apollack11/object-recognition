#!/usr/local/bin/python2.7
import argparse as ap
import cv2
import numpy as np
import os
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from scipy.cluster.vq import vq, kmeans, whiten
import time

def resize_image(image_file):
    img = cv2.imread(image_file)
    height,width = img.shape[:2]
    aspect_ratio = float(width)/height

    if width > 640:
        width_new = 640
        height_new = int(width_new / aspect_ratio)
        img = cv2.resize(img, (width_new, height_new), interpolation = cv2.INTER_CUBIC)
    return img

start = time.time()

# Load the classifier, class names, scaler, number of clusters and vocabulary
classifier, class_names, std_slr, k, vocabulary = joblib.load("trained_variables.pkl")

# Get the path of the testing set
parser = ap.ArgumentParser()
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("-t", "--testset", help="Path to test set")
group.add_argument("-i", "--image", help="Path to image")
parser.add_argument('-v',"--visualize", action='store_true')
args = vars(parser.parse_args())

# Get the path of the testing image(s) and store them in a list
image_paths = []
if args["testset"]:
    test_directory = args["testset"]
    try:
        os.listdir(test_directory)
    except OSError:
        print "No such directory {}\nCheck if the file exists".format(test_directory)
        exit()

    image_files = []
    for root, dirs, files in os.walk(test_directory, topdown=False):
        for d in dirs:
            for f in os.listdir(os.path.join(root,d)):
                if f.endswith('.jpg') or f.endswith('.JPG') or f.endswith('.png') or f.endswith('.jpeg'):
                    image_files.append(root + '/' + d + '/' + f)
else:
    image_files = [args["image"]]

# Create SIFT object
sift = cv2.SIFT()

# List where all the descriptors are stored
descriptor_list = []
for image_file in image_files:
    image = resize_image(image_file) # resizes image if it's larger than 640x480 or 480x640
    h, w = image.shape[:2]
    print ('Image size: ' + str(w) + ' x ' + str(h))
    # image = cv2.imread(image_file)
    kp, des = sift.detectAndCompute(image, None)
    descriptor_list.append((image_file, des))

descriptors = descriptor_list[0][1]
for image_file, descriptor in descriptor_list[1:]:
    descriptors = np.vstack((descriptors, descriptor))

#
test_features = np.zeros((len(image_files), k), "float32")
for i in xrange(len(image_files)):
    words, distance = vq(descriptor_list[i][1], vocabulary)
    for w in words:
        test_features[i][w] += 1

# Perform Tf-Idf vectorization
nbr_occurences = np.sum( (test_features > 0) * 1, axis = 0)
idf = np.array(np.log((1.0 * len(image_files) + 1) / (1.0 * nbr_occurences + 1)), 'float32')

# Scale the features
test_features = std_slr.transform(test_features)

# Perform the predictions
predictions = [class_names[i] for i in classifier.predict(test_features)]

print predictions

end = time.time()

print(end - start)
