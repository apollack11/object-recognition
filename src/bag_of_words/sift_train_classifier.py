#!/usr/local/bin/python2.7
import argparse as ap
import cv2
import numpy as np
import os
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from scipy.cluster.vq import vq, kmeans, whiten
from sklearn.preprocessing import StandardScaler
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

train_directory = "dataset/train"

class_names = []
image_files = []
image_labels = []
class_id = 0
for root, dirs, files in os.walk(train_directory, topdown=False):
    for d in dirs:
        if not d.startswith('.'):
            class_names.append(d)
        num_files = 0
        for f in os.listdir(os.path.join(root,d)):
            if f.endswith('.jpg') or f.endswith('.JPG') or f.endswith('.png') or f.endswith('.jpeg'):
                image_files.append(root + '/' + d + '/' + f)
                num_files += 1
        image_labels += [class_id] * num_files
        class_id += 1

print "Class Names:", class_names
print "Number of Image Files", len(image_files)

# Create SIFT object
sift = cv2.SIFT()

# List where all the descriptors are stored
print "Creating descriptor list"
start = time.time()

descriptor_list = []
for image_file in image_files:
    print image_file
    # image = cv2.imread(image_file)
    image = resize_image(image_file)
    kp, des = sift.detectAndCompute(image, None)
    descriptor_list.append((image_file, des))

end = time.time()
print "Creating descriptor list time:",(end - start)

print "Creating descriptors"
start = time.time()

descriptors = descriptor_list[0][1]
for image_file, descriptor in descriptor_list[1:]:
    if descriptor is not None:
        descriptors = np.vstack((descriptors, descriptor))

end = time.time()
print "Creating descriptors time:",(end - start)

# Perform k-means clustering
print "K-Means Clustering"
start = time.time()

descriptors = whiten(descriptors)

k = 100
vocabulary, variance = kmeans(descriptors, k, 1)

end = time.time()
print "K-means clustering time:",(end - start)

# Calculate the histogram of features
print "Creating histogram of features"
start = time.time()

features = np.zeros((len(image_files), k), "float32")
for i in xrange(len(image_files)):
    words, distance = vq(descriptor_list[i][1], vocabulary)
    for w in words:
        features[i][w] += 1

end = time.time()
print "Creating histogram of features time:",(end - start)

# Perform Tf-Idf vectorization
print "SVM setup"
start = time.time()

nbr_occurences = np.sum( (features > 0) * 1, axis = 0)
idf = np.array(np.log((1.0 * len(image_files) + 1) / (1.0 * nbr_occurences + 1)), 'float32')

# Scaling the words
std_slr = StandardScaler().fit(features)
features = std_slr.transform(features)

end = time.time()
print "SVM setup time:",(end - start)

# Train the Linear SVM
print "SVM training"
start = time.time()

classifier = LinearSVC()
classifier.fit(features, np.array(image_labels))

end = time.time()
print "SVM training time:",(end - start)

# Save the SVM
joblib.dump((classifier, class_names, std_slr, k, vocabulary), "trained_variables.pkl", compress=3)
